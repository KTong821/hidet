import torch
from hidet.apps.llm.builder import create_llm
from hidet.apps.llm.sampler import SamplingParams

llm = create_llm(name="meta-llama/Llama-2-7b-chat-hf")

sampling_params = SamplingParams(temperature=1e-6)
llm.add_sequence(0, "what is the meaning of life", sampling_params)
llm.step()
outputs = llm.step()

for output in outputs:
    print(output.prompt)
    print(output.sequence_id)
    print(output.output_text)
    print(output.status)
    print(output)