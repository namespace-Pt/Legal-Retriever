import os
from tqdm import tqdm
from loguru import logger
from typing import Optional, Union, List
from string import Formatter
from torch.utils.data import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PopulateTemplateCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, prefix: List[int], suffix: List[int]):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix = prefix
        self.suffix = suffix

    def __call__(self, batch_inputs):
        encoded_inputs = self.tokenizer(batch_inputs, add_special_tokens=False, max_length=self.max_length, truncation=True).input_ids
        lengths = [0 for _ in encoded_inputs]
        for i, encoded_input in enumerate(encoded_inputs):
            encoded_inputs[i] = self.prefix + encoded_input + self.suffix
            lengths[i] = len(encoded_inputs[i])
        outputs = {
            "text": self.tokenizer.batch_decode(encoded_inputs),
            "length": lengths
        }
        return outputs


class Templater:
    """
    Utility to apply template to the input text.
    
    - handle special tokens, like eos, and embed token
    - handle truncation w.r.t. max_length
    """
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.special_tokens = {"{text}": "<|standalone_text_token|>", "{eos}": tokenizer.eos_token, "{embed}": "<|standalone_text_embedding_token|>"}
        # NOTE: we manually add these special tokens to the tokenizer, which will be used when applying the template (do not)
        self.tokenizer.add_tokens(list(t for t in self.special_tokens.values() if t != tokenizer.eos_token))
        self.special_token_ids = {k: self.tokenizer.convert_tokens_to_ids(v) for k, v in self.special_tokens.items()}

        self.query_miss_counter = {}
        self.key_miss_counter = {}
    
    @property
    def text_token_id(self):
        return self.special_token_ids["{text}"]

    def get_template(self, query_template: Optional[str] = None, key_template: Optional[str] = None, dataset: Optional[str] = "msmarco"):
        """
        Get the template string for a given dataset.

        Args:
            query_template: a format string or a template name in QUERY_TEMPLATES.
            key_template: a format string or a template name in KEY_TEMPLATES.
            dataset: the name of the dataset.
        
        Template format:
            Mandatory:
                - There must be a {text} field in the template string, which will be populated with the input text.
            Optional: 
                - {eos} will be populated with the tokenizer's eos_token.
                - {embed} will be populated with a newly added special token (<|generate_text_embedding|>).
        """
        results = ()

        if query_template is not None:
            if "{text}" in query_template:
                results += (query_template,)
            else:
                assert query_template in QUERY_TEMPLATES, f"Unknown query template name: {query_template}"
                dataset_2_template = QUERY_TEMPLATES[query_template]
                if dataset in dataset_2_template:
                    template = dataset_2_template[dataset]
                else:
                    # use the default template
                    template = dataset_2_template[""]
                    miss = f"{query_template}--{dataset}"
                    if miss not in self.query_miss_counter:
                        logger.warning(f"No matching query template found for dataset [{dataset}] under [{query_template}], use the default template [{template}] instead.")
                        self.query_miss_counter[miss] = 1
                results += (template,)

        if key_template is not None:
            if "{text}" in key_template:
                results += (key_template,)
            else:
                assert key_template in KEY_TEMPLATES, f"Unknown key template name: {key_template}"
                dataset_2_template = KEY_TEMPLATES[key_template]
                if dataset in dataset_2_template:
                    template = dataset_2_template[dataset]
                else:
                    # use the default template
                    template = dataset_2_template[""]
                    miss = f"{key_template}--{dataset}"
                    if miss not in self.key_miss_counter:
                        logger.warning(f"No matching key template found for dataset [{dataset}] under [{key_template}], use the default template [{template}] instead.")
                        self.key_miss_counter[miss] = 1
                results += (template,)
        return results
    
    def apply(
        self, 
        inputs: Union[str, List[str]], 
        query_template: Optional[str] = None, 
        key_template: Optional[str] = None, 
        dataset: Optional[str] = None, 
        max_length: Optional[int] = None,
        # NOTE: encode in batch is faster
        batch_size: Optional[int] = None,
        show_progress: Optional[bool] = False,
        return_length: Optional[bool] = False,
    ):
        """
        Apply template on the inputs.

        Args:
            max_length: if specified, the inputs will be truncated to max_length - template_length before applying the template.
            return_length: if True, will return the length of the inputs after templating
        """

        if isinstance(inputs, str):
            inputs = [inputs]
            is_single = True
        else:
            is_single = False

        if batch_size is None:
            batch_size = len(inputs)

        # 1. get the template string
        template, = self.get_template(query_template=query_template, key_template=key_template, dataset=dataset)

        # 2. extract {text} and optional fields ({eos}, {embed}) from the template string
        fields = ["{" + field + "}" for _, field, _, _ in Formatter().parse(template) if field]
        assert "{text}" in fields, f"No text field found in the template: {template}"

        # 3. replace {text} and optional fields ({eos}, {embed}) with special tokens
        for field in fields:
            assert field in self.special_tokens, f"Unsupported field: {field}"
            template = template.replace(field, self.special_tokens[field])

        # 4. tokenize the template and the input queries
        encoded_template = self.tokenizer.encode(template, add_special_tokens=False)
        if max_length is not None:
            # NOTE: modify the max_length so that the templated inputs won't exceed max_length
            max_length = max_length - len(encoded_template) + 1
            assert max_length > 0, f"Make sure the max_length is bigger than the template_length - 1 = {len(encoded_template) - 1}!"

        # 5. locate the {text} field from the template
        # NOTE: search {text} from end to start
        for text_token_idx in range(len(encoded_template) - 1, -1, -1):
            if encoded_template[text_token_idx] == self.text_token_id:
                break


        if return_length:
            lengths = [0 for _ in inputs]

        # 6. populate the {text} field
        if len(inputs) <= batch_size:
            encoded_inputs = self.tokenizer(inputs, add_special_tokens=False, max_length=max_length, truncation=True).input_ids
            for i, encoded_input in enumerate(encoded_inputs):
                encoded_input = encoded_template[:text_token_idx] + encoded_input + encoded_template[text_token_idx + 1:]
                inputs[i] = self.tokenizer.decode(encoded_input)
                if return_length:
                    lengths[i] = len(encoded_input)

        else:
            # Use data collator for fast processing (faster than datasets or dataloader)
            dataloader = DataLoader(
                inputs, 
                batch_size=batch_size, 
                shuffle=False, 
                drop_last=False,
                pin_memory=False,
                collate_fn=PopulateTemplateCollator(
                    tokenizer=self.tokenizer,
                    max_length=max_length,
                    prefix=encoded_template[:text_token_idx],
                    suffix=encoded_template[text_token_idx + 1:],
                ),
                num_workers=16,
            )
            inputs = []
            for x in tqdm(dataloader, desc="Applying Template", disable=not show_progress):
                inputs.extend(x["text"])
                if return_length:
                    lengths.extend(x["length"])

        if is_single:
            inputs = inputs[0]
            if return_length:
                lengths = lengths[0]

        if return_length:
            return inputs, lengths
        else:
            return inputs


########## Query/Key Templates ##########
# 1. Each template must have a {text} field
# 2. Each template corresponds to a list of datasets
# 3. The template which includes "" in the datasets is used as the default one when no matching template is found
# 4. {eos} will be populated with the tokenizer's eos_token
# 5. {embed} will be populated with a newly added special token <|standalone_text_embedding_token|>


GROUPED_QUERY_TEMPLATES = {
    "no": [
        {
            "template": "{text}",
            "datasets": [""],
        }
    ],

    "v0": [
        # Retrieval
        {
            "template": "Given the query, retrieve relevant documents.\nQuery: {text}{eos}", 
            "datasets": [""]
        },
        # Others
        {
            "template": "{text}{eos}", 
            "datasets": []
        },
    ],

    "v1": [
        # Retrieval
        {
            "template": "Query: {text}\nUse one word to summarize the query's relevant information. The word is: \"", 
            "datasets": [""]
        }
    ],
}

GROUPED_KEY_TEMPLATES = {
    "no": [
        {
            "template": "{text}",
            "datasets": [""],
        }
    ],

    "v0": [
        {
            "template": "{text}{eos}", 
            "datasets": [""]
        },
    ],

    "v1": [
        {
            "template": "Text: {text}\nUse one word to summarize the text's content. The word is: \"", 
            "datasets": [""]
        },
    ],
}


# NOTE: transform the query/key templates into dictionary of {dataset: template} to speed up finding template
QUERY_TEMPLATES = {}
for template_name, templates in GROUPED_QUERY_TEMPLATES.items():
    has_default = False
    flattened_template = {}
    for item in templates:
        template = item["template"]
        datasets = item["datasets"]
        for dataset in datasets:
            if dataset == "":
                if has_default:
                    raise ValueError(f"Found multiple default templates that include the empty dataset \"\"! Make sure only one template includes dataset \"\"!")
                has_default = True
            flattened_template[dataset] = template
    if not has_default:
        raise ValueError(f"No default template found under {template_name}! Make sure one template includes dataset \"\" so that it becomes the default one when no matching templates are found.")
    QUERY_TEMPLATES[template_name] = flattened_template

KEY_TEMPLATES = {}
for template_name, templates in GROUPED_KEY_TEMPLATES.items():
    has_default = False
    flattened_template = {}
    for item in templates:
        template = item["template"]
        datasets = item["datasets"]
        for dataset in datasets:
            if dataset == "":
                if has_default:
                    raise ValueError(f"Found multiple default templates that include the empty dataset \"\"! Make sure only one template includes dataset \"\"!")
                has_default = True
            flattened_template[dataset] = template
    if not has_default:
        raise ValueError(f"No default template found under {template_name}! Make sure one template includes dataset \"\" so that it becomes the default one when no matching templates are found.")
    KEY_TEMPLATES[template_name] = flattened_template
