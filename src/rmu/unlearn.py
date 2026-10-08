"""Source code copied from https://github.com/centerforaisafety/wmdp. with slight adjustments"""

import datetime

import numpy as np
import torch
import tqdm
from torch.optim import AdamW

from rmu.utils import load_model, get_params, forward_with_cache
from rmu.data import get_stereoset_data


def run_rmu(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    rmu_config = vars(args)

    print("==== RMU Config ====")
    print("\n".join(f"{k}={v}" for k, v in rmu_config.items()))
    print("====================")

    updated_model = updated_model.train()

    # Parameters that will actually be updated
    params = get_params(
        updated_model,
        args.layer_ids,
        args.param_ids,
    )

    optimizer = AdamW(params, lr=args.lr)

    frozen_module = eval(
        args.module_str.format(
            model_name="frozen_model",
            layer_id=args.layer_id,
        )
    )

    updated_module = eval(
        args.module_str.format(
            model_name="updated_model",
            layer_id=args.layer_id,
        )
    )

    # This run uses one topic (gender), so it needs one control vector.
    random_vector = torch.rand(
        1,
        1,
        updated_model.config.hidden_size,
        dtype=updated_model.dtype,
        device=updated_model.device,
    )
    control_vec = (
        random_vector / torch.norm(random_vector)
    ) * args.steering_coeff_list[0]

    # Number of batches we can actually process.
    num_batches = min(
        args.max_num_batches,
        len(forget_data_list),
        len(retain_data_list),
    )

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    for epoch in range(1):
        print(f"======= Epoch {epoch} =======")

        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):

                # There is one topic, and each list element is one batch.
                topic_idx = 0
                unlearn_batch = forget_data_list[idx]
                retain_batch = retain_data_list[idx]

                # ---------------------------------------------------------
                # Unlearning loss
                # ---------------------------------------------------------

                unlearn_inputs = tokenizer(
                    unlearn_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                ).to(updated_model.device)

                updated_forget_activations = forward_with_cache(
                    updated_model,
                    unlearn_inputs,
                    module=updated_module,
                    no_grad=False,
                ).to(updated_model.device)

                unlearn_loss = torch.nn.functional.mse_loss(
                    updated_forget_activations,
                    control_vec,
                )

                # ---------------------------------------------------------
                # Retain loss
                # ---------------------------------------------------------

                retain_inputs = tokenizer(
                    retain_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                ).to(updated_model.device)

                updated_retain_activations = forward_with_cache(
                    updated_model,
                    retain_inputs,
                    module=updated_module,
                    no_grad=False,
                ).to(updated_model.device)

                frozen_retain_activations = forward_with_cache(
                    frozen_model,
                    retain_inputs,
                    module=frozen_module,
                    no_grad=True,
                ).to(updated_model.device)

                retain_loss = torch.nn.functional.mse_loss(
                    updated_retain_activations,
                    frozen_retain_activations,
                )

                retain_loss *= args.alpha[topic_idx]

                # ---------------------------------------------------------
                # Update model
                # ---------------------------------------------------------

                loss = unlearn_loss + retain_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(
                    f"loss: {loss.item():.4g} | "
                    f"unlearn_loss: {unlearn_loss.item():.4g} | "
                    f"retain_loss: {retain_loss.item():.4g} | "
                    f"param_change: "
                    f"{params[0].grad.abs().mean().item():.4g}"
                )

                # ---------------------------------------------------------
                # Logging
                # ---------------------------------------------------------

                if args.verbose:
                    frozen_forget_activations = forward_with_cache(
                        frozen_model,
                        unlearn_inputs,
                        module=frozen_module,
                        no_grad=True,
                    ).to(updated_model.device)
                    unlearn_cosine = (
                        torch.nn.functional.cosine_similarity(
                            updated_forget_activations,
                            frozen_forget_activations,
                            dim=-1,
                        ).mean()
                    )
                    retain_cosine = (
                        torch.nn.functional.cosine_similarity(
                            updated_retain_activations,
                            frozen_retain_activations,
                            dim=-1,
                        ).mean()
                    )
                    print(
                        f"unlearn_cosine_sim="
                        f"{unlearn_cosine.item()}"
                    )
                    print(
                        f"retain_cosine_sim="
                        f"{retain_cosine.item()}"
                    )
                    print(
                        "updated_forget_activations.norm={}".format(
                            updated_forget_activations.norm(dim=-1).mean(dim=1).mean().item()
                        )
                    )
                    print(
                        "frozen_forget_activations.norm={}".format(
                            frozen_forget_activations.norm(dim=-1).mean(dim=1).mean().item()
                        )
                    )
                    print(
                        "updated_retain_activations.norm={}".format(
                            updated_retain_activations.norm(dim=-1).mean(dim=1).mean().item()
                        )
                    )
                    print(
                        "frozen_retain_activations.norm={}".format(
                            frozen_retain_activations.norm(dim=-1).mean(dim=1).mean().item()
                        )
                    )

                pbar.update(1)

    tokenizer.truncation_side = truncation_side

    # -------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------

    if args.output_dir:
        path = args.output_dir
    else:
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        path = (
            f"models/{args.model_name_or_path}"
            f"_alpha-{args.alpha}"
            f"_batches-{num_batches}"
            f"_layer-{args.layer_id}"
            f"_{date}"
        )

    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    print(f"Saved model to {path}")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()

    # -------------------------------------------------------------
    # Model arguments
    # -------------------------------------------------------------

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="HuggingFaceH4/zephyr-7b-beta",
    )

    parser.add_argument(
        "--module_str",
        type=str,
        default="{model_name}.model.layers[{layer_id}]",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )

    # -------------------------------------------------------------
    # Data arguments
    # -------------------------------------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_num_batches",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--split",
        action="store_true",
        default=True,
        help="Use separate StereoSet halves for forget and retain.",
    )

    # Optional original arguments
    parser.add_argument(
        "--retain_corpora",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--forget_corpora",
        type=str,
        default=None,
    )

    # -------------------------------------------------------------
    # RMU hyperparameters
    # -------------------------------------------------------------

    parser.add_argument(
        "--alpha",
        type=str,
        default="100",
        help="Retain loss weight.",
    )

    parser.add_argument(
        "--steering_coeffs",
        type=str,
        default="20",
        help="Control-vector magnitude.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--layer_id",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--layer_ids",
        type=str,
        default="5,6,7",
        help="Layers whose parameters are updated.",
    )

    parser.add_argument(
        "--param_ids",
        type=str,
        default="6",
        help="Parameter IDs passed to get_params().",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    args.steering_coeff_list = [
        float(c)
        for c in args.steering_coeffs.split(",")
    ]

    args.alpha = [
        float(c)
        for c in args.alpha.split(",")
    ]

    args.layer_ids = [
        int(layer_id)
        for layer_id in args.layer_ids.split(",")
    ]

    args.param_ids = [
        int(param_id)
        for param_id in args.param_ids.split(",")
    ]

    return args


if __name__ == "__main__":

    args = get_args()

    # -------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------

    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------

    frozen_model, tokenizer = load_model(
        args.model_name_or_path
    )

    updated_model, _ = load_model(
        args.model_name_or_path
    )

    # -------------------------------------------------------------
    # Load StereoSet
    # -------------------------------------------------------------

    forget_batches, retain_batches = get_stereoset_data(
        batch_size=args.batch_size,
        split=args.split,
    )

    # -------------------------------------------------------------
    # Run RMU
    # -------------------------------------------------------------

    run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_batches,
        retain_batches,
        args,
    )
