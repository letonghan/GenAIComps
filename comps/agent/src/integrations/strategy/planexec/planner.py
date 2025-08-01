# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import operator
from typing import Annotated, List, Sequence, Tuple, TypedDict

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from comps.cores.telemetry.opea_telemetry import opea_telemetry, tracer

from ...global_var import threads_global_kv
from ...utils import has_multi_tool_inputs, tool_renderer
from ..base_agent import BaseAgent
from .prompt import (
    answer_check_prompt,
    answer_make_prompt,
    hwchase17_react_prompt,
    plan_check_prompt,
    planner_prompt,
    replanner_prompt,
)


class PlanExecute(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    output: str


class Plan(BaseModel):
    """Plan to follow in future."""

    steps: List[str] = Field(description="different steps to follow, should be in sorted order")


class Response(BaseModel):
    """Response to user."""

    response: str


class PlanStepChecker:
    """Determines whether this plan making sense or not.

    Returns:
        str: A decision for whether we should use this plan or not
    """

    @opea_telemetry
    def __init__(self, llm, is_vllm=False):
        class grade(BaseModel):
            binary_score: str = Field(description="executable score 'yes' or 'no'")

        if is_vllm:
            llm = llm.bind_tools([grade], tool_choice={"function": {"name": grade.__name__}})
        else:
            llm = llm.bind_tools([grade])

        output_parser = PydanticToolsParser(tools=[grade], first_tool_only=True)
        self.chain = plan_check_prompt | llm | output_parser

    @opea_telemetry
    def __llm_invoke__(self, state):
        scored_result = self.chain.invoke(state)
        return scored_result

    @opea_telemetry
    def __call__(self, state):
        # print("---CALL PlanStepChecker---")
        scored_result = self.__llm_invoke__(state)
        score = scored_result.binary_score
        print(f"Task is {state['context']}, Score is {score}")
        if score.startswith("yes"):
            return True
        else:
            return False


# Define workflow Node
class Planner:
    @opea_telemetry
    def __init__(self, llm, plan_checker=None, is_vllm=False):
        if is_vllm:
            llm = llm.bind_tools([Plan], tool_choice={"function": {"name": Plan.__name__}})
        else:
            llm = llm.bind_tools([Plan])
        output_parser = PydanticToolsParser(tools=[Plan], first_tool_only=True)
        self.llm = planner_prompt | llm | output_parser
        self.plan_checker = plan_checker

    @opea_telemetry
    def __llm_invoke__(self, messages):
        plan = self.llm.invoke(messages)
        return plan

    @opea_telemetry
    def __call__(self, state):
        print("---CALL Planner---")
        input = state["messages"][-1].content
        success = False
        # sometime, LLM will not provide accurate steps per ask, try more than one time until success
        while not success:
            while not success:
                try:
                    plan = self.__llm_invoke__({"messages": [("user", state["messages"][-1].content)]})
                    print("Generated plan: ", plan)
                    success = True
                except OutputParserException as e:
                    pass
                except Exception as e:
                    raise e

            steps = []
            for step in plan.steps:
                if self.plan_checker({"context": step, "question": input}):
                    steps.append(step)

            if len(steps) == 0:
                success = False
        print("Steps: ", steps)
        return {"input": input, "plan": steps}


class Executor:
    @opea_telemetry
    def __init__(self, llm, tools=[]):
        prompt = hwchase17_react_prompt
        if has_multi_tool_inputs(tools):
            raise ValueError("Only supports single input tools when using strategy == react")
        else:
            agent_chain = create_react_agent(llm, tools, prompt, tools_renderer=tool_renderer)
        self.agent_executor = AgentExecutor(
            agent=agent_chain, tools=tools, handle_parsing_errors=True, max_iterations=50
        )

    @opea_telemetry
    def __call__(self, state):
        print("---CALL Executor---")
        plan = state["plan"]
        out_state = []
        for i, step in enumerate(plan):
            task_formatted = f"""
You are tasked with executing {step}.

You can leverage output from previous steps to help you.
previous steps and output: {out_state}
"""
            success = False
            print(task_formatted)
            while not success:
                agent_response = self.agent_executor.invoke({"input": task_formatted})
                output = agent_response["output"]
                success = True
            print(f"Task is {step}, Response is {output}")
            out_state.append(f"Task is {step}, Response is {output}")
        return {
            "past_steps": out_state,
        }


class AnswerMaker:
    @opea_telemetry
    def __init__(self, llm, is_vllm=False):
        if is_vllm:
            llm = llm.bind_tools([Response], tool_choice={"function": {"name": Response.__name__}})
        else:
            llm = llm.bind_tools([Response])
        output_parser = PydanticToolsParser(tools=[Response], first_tool_only=True)
        self.llm = answer_make_prompt | llm | output_parser

    @opea_telemetry
    def __llm_invoke__(self, state):
        output = self.llm.invoke(state)
        return output

    @opea_telemetry
    def __call__(self, state):
        print("---CALL AnswerMaker---")
        success = False
        # sometime, LLM will not provide accurate steps per ask, try more than one time until success
        while not success:
            try:
                output = self.__llm_invoke__(state)
                print("Generated response: ", output.response)
                success = True
            except OutputParserException as e:
                pass
            except Exception as e:
                raise e

        return {"output": output.response}


class FinalAnswerChecker:
    """Determines whether this final answer making sense or not.

    Returns:
        str: A decision for whether we should use this plan or not
    """

    @opea_telemetry
    def __init__(self, llm, is_vllm=False):
        class grade(BaseModel):
            binary_score: str = Field(description="executable score 'yes' or 'no'")

        if is_vllm:
            llm = llm.bind_tools([grade], tool_choice={"function": {"name": grade.__name__}})
        else:
            llm = llm.bind_tools([grade])
        output_parser = PydanticToolsParser(tools=[grade], first_tool_only=True)
        self.chain = answer_check_prompt | llm | output_parser

    @opea_telemetry
    def __llm_invoke__(self, state):
        output = self.chain.invoke(state)
        return output

    @opea_telemetry
    def __call__(self, state):
        print("---CALL FinalAnswerChecker---")
        scored_result = self.__llm_invoke__(state)
        score = scored_result.binary_score
        print(f"Answer is {state['response']}, Grade of good response is {score}")
        if score.startswith("yes"):
            return END
        else:
            return "replan"


class Replanner:
    @opea_telemetry
    def __init__(self, llm, answer_checker=None):
        llm = llm.bind_tools([Plan])
        output_parser = PydanticToolsParser(tools=[Plan], first_tool_only=True)
        self.llm = replanner_prompt | llm | output_parser
        self.answer_checker = answer_checker

    @opea_telemetry
    def __llm_invoke__(self, state):
        output = self.llm.invoke(state)
        return output

    @opea_telemetry
    def __call__(self, state):
        print("---CALL Replanner---")
        success = False
        # sometime, LLM will not provide accurate steps per ask, try more than one time until success
        while not success:
            try:
                output = self.__llm_invoke__(state)
                success = True
                print("Replan: ", output)
            except OutputParserException as e:
                pass
            except Exception as e:
                raise e

        return {"plan": output.steps}


class PlanExecuteAgentWithLangGraph(BaseAgent):
    @opea_telemetry
    def __init__(self, args, tools_descriptions=None, with_memory=False, **kwargs):
        super().__init__(args, tools_descriptions, local_vars=globals(), **kwargs)

        # Define Node
        plan_checker = PlanStepChecker(self.llm, is_vllm=self.is_vllm)

        plan_step = Planner(self.llm, plan_checker, is_vllm=self.is_vllm)
        execute_step = Executor(self.llm, self.tools_descriptions)
        make_answer = AnswerMaker(self.llm, is_vllm=self.is_vllm)

        # Define Graph
        workflow = StateGraph(PlanExecute)
        workflow.add_node("planner", plan_step)
        workflow.add_node("plan_executor", execute_step)
        workflow.add_node("answer_maker", make_answer)

        # Define edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "plan_executor")
        workflow.add_edge("plan_executor", "answer_maker")
        workflow.add_edge("answer_maker", END)

        if with_memory:
            self.app = workflow.compile(checkpointer=MemorySaver())
        else:
            self.app = workflow.compile()

    def prepare_initial_state(self, query):
        return {"messages": [("user", query)]}

    @opea_telemetry
    async def stream_generator(self, query, config, thread_id=None):
        initial_state = self.prepare_initial_state(query)
        if thread_id is not None:
            config["configurable"] = {"thread_id": thread_id}
        async for event in self.app.astream(initial_state, config=config):
            if thread_id is not None:
                with threads_global_kv as g_threads:
                    thread_inst, created_at, status = g_threads[thread_id]
                    if status == "try_cancel":
                        yield "[thread_completion_callback] signal to cancel! Changed status to ready"
                        print("[thread_completion_callback] signal to cancel! Changed status to ready")
                        g_threads[thread_id] = (thread_inst, created_at, "ready")
                        break
            for node_name, node_state in event.items():
                yield f"--- CALL {node_name} ---\n"
                for k, v in node_state.items():
                    if v is not None:
                        yield f"{k}: {v}\n"

            yield f"data: {repr(event)}\n\n"
        yield "data: [DONE]\n\n"

    @opea_telemetry
    async def non_streaming_run(self, query, config):
        initial_state = self.prepare_initial_state(query)
        try:
            async for s in self.app.astream(initial_state, config=config, stream_mode="values"):
                for k, v in s.items():
                    print(f"{k}: {v}\n")

            last_message = s["output"]
            print("******Response: ", last_message)
            return last_message
        except Exception as e:
            return str(e)
