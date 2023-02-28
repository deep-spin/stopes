import pickle
import argparse
import os
import torch
import pandas as pd

from tqdm import tqdm
from stopes.eval.alti.wrappers.transformer_wrapper import FairseqTransformerHub
from stopes.eval.alti.wrappers.multilingual_transformer_wrapper import FairseqMultilingualTransformerHub
from stopes.eval.alti.alti_metrics.alti_metrics_utils import compute_alti_nllb, compute_alti_metrics

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def main(args):
    # Checkpoint path
    model_dict = args.model_dict
    ckpt_path = args.ckpt_path

    source_lang = args.source_lang
    target_lang = args.target_lang

    # checkpoint_dir is the folder where the checkpoint is located
    # checkpoint_file is the name of the checkpoint file
    checkpoint_dir = os.path.dirname(ckpt_path)
    checkpoint_file = os.path.basename(ckpt_path)

    # the checkpoint file finishes with "model_last_checkpoint.pt"; obtain the name of the model
    model_name = checkpoint_file.split("_")[0]

    # load the model, vocabulary and the sentencepiece tokenizer
    hub = FairseqMultilingualTransformerHub.from_pretrained(
        checkpoint_dir=checkpoint_dir,
        checkpoint_file=checkpoint_file,
        data_name_or_path=args.data_path,
        source_lang=source_lang,
        target_lang=target_lang,
        lang_pairs=f'{source_lang}-{target_lang}',
        fixed_dictionary=model_dict,
    )

    dataset_original = pd.read_pickle(args.df_path)
    if args.perturbation_file:
        dataset_original = dataset_original.loc[dataset_original["candidate"]==1]
        dataset_original = dataset_original.reset_index()
        dataset_original = dataset_original.loc[dataset_original.model == model_name]

    alti_scores = []
    for i in tqdm(dataset_original.index):
        src_tensor = torch.tensor(dataset_original.loc[i, 'src_ids'], device=hub.device)
        tgt_tensor = torch.tensor(dataset_original.loc[i, 'mt_ids'][:-1], device=hub.device)

        # Add id 2 to the beginning of the target sentence
        tgt_tensor = torch.cat((torch.tensor([2], device=hub.device), tgt_tensor))

        src_sent = dataset_original.loc[i, 'src']
        tgt_sent = dataset_original.loc[i, 'mt']
        
        attributions, src_tok, tgt_tok, pred_tok = compute_alti_nllb(hub, src_tensor, tgt_tensor)
        metrics = compute_alti_metrics(attributions, src_tok, tgt_tok, pred_tok)
        alti = metrics['avg_sc']
        alti_scores.append({"alti": alti, "src": src_sent, "tgt": tgt_sent})

    # Add alti scores to dataset
    dataset_original['alti'] = [alti_scores[i]['alti'] for i in range(len(alti_scores))]

    print(f"Saving alti scores for model {model_name} to pickle file.")
    # Save alti scores to pickle file in data_path/alti/model_name folder; create folder if it doesn't exist
    alti_path = os.path.join(args.data_path, "alti", model_name)
    if not os.path.exists(alti_path):
        os.makedirs(alti_path)
    dataset_original.to_pickle(os.path.join(alti_path, "df_w_alti.pkl"))

    # Print 25, median and 75 percentiles
    print(dataset_original['alti'].describe(percentiles=[.25, .5, .75]))
    # Save that info to a text file
    with open(os.path.join(args.data_path, "alti", model_name, "alti_scores.txt"), "w") as f:
        f.write(dataset_original['alti'].describe(percentiles=[.25, .5, .75]).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--df_path", type=str, required=True)
    parser.add_argument("--source_lang", type=str, required=True)
    parser.add_argument("--target_lang", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--perturbation_file", type=bool, default=False, help="If True, perturbation file is used to compute alti scores")
    parser.add_argument("--model_dict", type=str, default="/home/nunomg/llm-hallucination/fairseq/model_dict.128k.txt")
    args = parser.parse_args()
    main(args)
