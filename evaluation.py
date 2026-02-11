#############################
# reference: https://github.com/explodinggradients/ragas/pull/2045/files#diff-d94328b97d297c25198ab4a75969cd8fb7463215540cf6436f8d602a4fb8b22b
# RAGAS PARSER PATCH:
import sys
import ragas
import torch
module = sys.modules["ragas"]
from langchain_core.exceptions import OutputParserException
from ragas.callbacks import new_group
from ragas.exceptions import RagasOutputParserException
from ragas.prompt.utils import extract_json
from json_repair import repair_json
from ragas.prompt.pydantic_prompt import fix_output_format_prompt

async def parse_output_string(
    self,
    output_string: str,
    prompt_value,
    llm,
    callbacks,
    retries_left: int = 3,
):
    callbacks = callbacks or []
    try:
        # First attempt to extract JSON from the output
        jsonstr = extract_json(output_string)
        jsonstr = repair_json(jsonstr)
        result = super(type(self), self).parse(jsonstr)
    except OutputParserException as e:
        # If JSON extraction fails, try more aggressive parsing
        if retries_left != 0:
            retry_rm, retry_cb = new_group(
                name="fix_output_format",
                inputs={"output_string": output_string},
                callbacks=callbacks,
            )
            
            # Add more explicit instructions for the fix_output_format prompt
            fixed_output_string = await fix_output_format_prompt.generate(
                llm=llm,
                data=module.prompt.pydantic_prompt.OutputStringAndPrompt(
                    output_string=output_string,
                    prompt_value=prompt_value.to_string(),
                ),
                callbacks=retry_cb,
                retries_left=retries_left - 1,
            )
            retry_rm.on_chain_end({"fixed_output_string": fixed_output_string})

            fixed_jsonstr = extract_json(fixed_output_string.text)
            fixed_jsonstr = repair_json(fixed_jsonstr)
            result = super(type(self), self).parse(fixed_jsonstr)
        else:
            raise RagasOutputParserException()
    return result

module.prompt.pydantic_prompt.RagasOutputParser.parse_output_string = parse_output_string
sys.modules["ragas"] = module
#############################

from datasets import Dataset
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from ragas.llms.base import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from llama_index.llms.openai import OpenAI


gt_map = {
    'SUPPORTS': "The claim in this statement is correct.", 
    'REFUTES': "The claim in this statement is wrong.", 
    'NOTENOUGHINFO': "There is not enough information about the claim in this statement."
}

def evaluate_rag(df, dataset_name):
    df = df.fillna('')
    filtered_df = df.loc[:, ["question", "response", "contexts", "gt_response"]].copy()
    filtered_df = filtered_df.rename(columns={'response': 'answer', 'gt_response': 'ground_truth'})
    
    if dataset_name in ["fever"]:
        filtered_df["ground_truth"] = filtered_df["ground_truth"].map(gt_map)

    data = {column:filtered_df[column].to_list() for column in filtered_df.columns}
    data["contexts"] = [eval(item.replace("\'\n \'", "\' , \'")) for item in data["contexts"]]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    llm = OpenAI(model="gpt-4o-mini")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    llm_wrapper = LlamaIndexLLMWrapper(llm)
    embed_wrapper = LlamaIndexEmbeddingsWrapper(embed_model)

    print("Running RAGAS evaluation on FEVER dataset...")
    METRICS = [
        answer_relevancy,
        faithfulness,
    ]

    result = evaluate(
        llm=llm_wrapper,
        embeddings=embed_wrapper,
        dataset=Dataset.from_dict(data),
        metrics=METRICS,
        batch_size=64,
        raise_exceptions=False,
    )
    print(result)