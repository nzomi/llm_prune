# Import necessary modules
import time
import torch
import torch.nn as nn
from tqdm import tqdm

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

from collections import defaultdict
import fnmatch


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        # ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
        ppl_test = eval_ppl_wikitext(model, testloader, device=device)
    return ppl_test 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
# def eval_ppl_wikitext(model, testenc, bs=1, device=None):
#     # Get input IDs
#     testenc = testenc.input_ids

#     # Calculate number of samples
#     nsamples = testenc.numel() // model.seqlen

#     # List to store negative log likelihoods
#     nlls = []
#     print(f"nsamples {nsamples}")

#     # Loop through each batch
#     for i in tqdm(range(0,nsamples,bs)):
#         if i % 50 == 0:
#             print(f"sample {i}")

#         # Calculate end index
#         j = min(i+bs, nsamples)

#         # Prepare inputs and move to device
#         inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
#         inputs = inputs.reshape(j-i, model.seqlen)

#         # Forward pass through the model
#         lm_logits = model(inputs).logits

#         # Shift logits and labels for next token prediction
#         shift_logits = lm_logits[:, :-1, :].contiguous()
#         shift_labels = inputs[:, 1:]

#         # Compute loss
#         loss_fct = nn.CrossEntropyLoss()
#         loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

#         # Calculate negative log likelihood
#         neg_log_likelihood = loss.float() * model.seqlen * (j-i)

#         # Append to list of negative log likelihoods
#         nlls.append(neg_log_likelihood)

#     # Compute perplexity
#     ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

#     # Empty CUDA cache to save memory
#     torch.cuda.empty_cache()

#     return ppl.item()

def eval_ppl_wikitext(model, testenc, max_length=12288, stride=2048, device=None):
    input_ids = testenc.input_ids[0].to(device)
    seq_len = input_ids.size(0)
    model = model.to(device)
    nlls = []
    total_tokens = 0

    loss_fct = nn.CrossEntropyLoss(reduction='sum')  # sum over tokens

    for start_idx in tqdm(range(0, seq_len - 1, stride)):
        end_idx = min(start_idx + max_length, seq_len)

        input_ids_slice = input_ids[start_idx:end_idx].unsqueeze(0).to(device)

        # shift labels for next token prediction
        labels = input_ids_slice.clone()
        
        with torch.no_grad():
            outputs = model(input_ids_slice, labels=labels)
            neg_log_likelihood = outputs.loss * (end_idx - start_idx)

        nlls.append(neg_log_likelihood)
        total_tokens += end_idx - start_idx

        if end_idx == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    return ppl.item()

def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer, 
        add_special_tokens=add_special_tokens
    )

    return results 