# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from fastapi.responses import StreamingResponse
from huggingface_hub import AsyncInferenceClient
from langsmith import traceable

from comps import (
    GeneratedDoc,
    LLMParamsDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

prompt_1024 = """
### You are a helpful, respectful and honest assistant to help the user with questions. \
Please refer to the search results obtained from the local knowledge base. \
But be careful to not incorporate the information that you think is not relevant to the question. \
If you don't know the answer to a question, please don't share false information. \
### Search results: In a world where technology has advanced beyond our wildest dreams, humanity stands on the brink of a new era. The year is 2050, and artificial intelligence has become an integral part of everyday life. Autonomous vehicles zip through the streets, drones deliver packages with pinpoint accuracy, and smart homes anticipate every need of their inhabitants. But with these advancements come new challenges and ethical dilemmas. As society grapples with the implications of AI, questions about privacy, security, and the nature of consciousness itself come to the forefront. Amidst this backdrop, a new breakthrough in quantum computing promises to revolutionize the field even further. Scientists have developed a quantum processor capable of performing calculations at speeds previously thought impossible. This leap in technology opens the door to solving problems that have long stumped researchers, from predicting climate change patterns with unprecedented accuracy to unraveling the mysteries of the human genome. However, the power of this new technology also raises concerns about its potential misuse. Governments and corporations race to secure their own quantum capabilities, sparking a new kind of arms race. Meanwhile, a group of rogue programmers, known as the Shadow Collective, seeks to exploit the technology for their own ends. As tensions rise, a young scientist named Dr. Evelyn Zhang finds herself at the center of this unfolding drama. She has discovered a way to harness quantum computing to create a true artificial general intelligence (AGI), a machine capable of independent thought and reasoning. Dr. Zhang's creation, named Athena, possesses the potential to either save humanity from its own worst impulses or to become the ultimate instrument of control. As she navigates the treacherous waters of corporate espionage, government intrigue, and ethical quandaries, Dr. Zhang must decide the fate of her creation and, with it, the future of humanity. Will Athena be a benevolent guardian or a malevolent dictator? The answer lies in the choices made by those who wield its power. The world watches with bated breath as the next chapter in the saga of human and machine unfolds. In the midst of these global tensions, everyday life continues. Children attend schools where AI tutors provide personalized learning experiences. Hospitals use advanced algorithms to diagnose and treat patients with greater accuracy than ever before. The entertainment industry is transformed by virtual reality experiences that are indistinguishable from real life. Yet, for all the benefits, there are those who feel left behind by this technological revolution. Communities that once thrived on traditional industries find themselves struggling to adapt. The digital divide grows wider, creating new forms of inequality. Dr. Zhang's journey is not just a scientific quest but a deeply personal one. Her motivations are shaped by a desire to honor her late father's legacy, a pioneer in the field of AI who envisioned a future where technology would serve humanity's highest ideals. As she delves deeper into her research, she encounters allies and adversaries from unexpected quarters. A former colleague, Dr. Marcus Holt, now working for a rival tech giant, becomes both a rival and a potential ally as they navigate their complex relationship. In a hidden lab, far from prying eyes, Dr. Zhang and her team work tirelessly to refine Athena. They face numerous setbacks and breakthroughs, each step bringing them closer to their goal. The ethical implications of their work weigh heavily on them. Can a machine truly understand human emotions? Is it possible to program empathy and compassion? These questions haunt Dr. Zhang as she watches Athena's capabilities grow. As word of Athena's development leaks, the world reacts with a mixture of hope and fear. Protests erupt in major cities, with demonstrators demanding transparency and ethical oversight. Governments convene emergency sessions to discuss the potential impact of AGI on national security and global stability. Amid the chaos, the Shadow Collective launches a cyber-attack on Dr. Zhang's lab, attempting to steal her research. The attack is thwarted, but it serves as a stark reminder of the dangers they face. The final phase of Athena's development involves a series of tests to evaluate her decision-making abilities. This is the whole story.\n
### Question: Summarize the story above into three sentences.\n
### Answer:
"""

@register_microservice(
    name="opea_service@llm_tgi",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9000,
)
@traceable(run_type="llm")
@register_statistics(names=["opea_service@llm_tgi"])
async def llm_generate(input: LLMParamsDoc):
    stream_gen_time = []
    start = time.time()
    if input.streaming:

        async def stream_generator():
            chat_response = ""
            text_generation = await llm.text_generation(
                prompt=prompt_1024,
                stream=input.streaming,
                max_new_tokens=input.max_new_tokens,
                repetition_penalty=input.repetition_penalty,
                temperature=input.temperature,
                top_k=input.top_k,
                top_p=input.top_p,
            )
            async for text in text_generation:
                stream_gen_time.append(time.time() - start)
                chat_response += text
                chunk_repr = repr(text.encode("utf-8"))
                print(f"[llm - chat_stream] chunk:{chunk_repr}")
                yield f"data: {chunk_repr}\n\n"
            print(f"[llm - chat_stream] stream response: {chat_response}")
            statistics_dict["opea_service@llm_tgi"].append_latency(stream_gen_time[-1], stream_gen_time[0])
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        response = await llm.text_generation(
            prompt=input.query,
            stream=input.streaming,
            max_new_tokens=input.max_new_tokens,
            repetition_penalty=input.repetition_penalty,
            temperature=input.temperature,
            top_k=input.top_k,
            top_p=input.top_p,
        )
        statistics_dict["opea_service@llm_tgi"].append_latency(time.time() - start, None)
        return GeneratedDoc(text=response, prompt=input.query)


if __name__ == "__main__":
    llm_endpoint = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")
    llm = AsyncInferenceClient(
        model=llm_endpoint,
        timeout=600,
    )
    opea_microservices["opea_service@llm_tgi"].start()
