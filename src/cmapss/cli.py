"""One CLI; imports torch only for commands which actually need it."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="C-MAPSS LSTM / DANN, no implicit downloads")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Validate one partition; no model dependency")
    inspect.add_argument("--data-dir", default="src")
    inspect.add_argument("--dataset", default="FD001")
    inspect.add_argument("--partition", choices=["train", "test"], default="train")
    for name in ["train", "adapt", "seeds"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        if name == "adapt":
            p.add_argument("--source-checkpoint", required=True)
        if name != "seeds":
            p.add_argument("--resume")
        else:
            p.add_argument("--seeds", type=int, nargs="+", required=True)
    for name in ["evaluate", "transfer"]:
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--data-dir", default="src")
        p.add_argument("--dataset", default="FD002" if name == "transfer" else "FD001")
        p.add_argument("--output", required=True)
        p.add_argument("--device", default="auto")
        p.add_argument("--no-trajectories", action="store_true")
    p = sub.add_parser("figures", help="Explicitly regenerate PNG figures from saved CSVs")
    p.add_argument("--run", required=True)
    p = sub.add_parser("synthetic", help="Generate local synthetic files explicitly")
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="FD001")
    p = sub.add_parser("smoke", help="Synthetic FD001-only training and evaluation, CPU")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        from .data import read_frames, terminal_labels
        df = read_frames(Path(args.data_dir) / f"{args.partition}_{args.dataset}.txt")
        result = {"dataset": args.dataset, "partition": args.partition,
                  "rows": len(df), "engines": int(df.unit_id.nunique()), "features": 24}
        if args.partition == "test":
            terminal_labels(Path(args.data_dir) / f"RUL_{args.dataset}.txt", df)
            result["terminal_mapping_verified"] = True
        print(json.dumps(result, indent=2))
    elif args.command == "synthetic":
        from .synthetic import generate
        generate(args.output, args.dataset)
    elif args.command == "figures":
        from .plots import training_figures, evaluation_figures
        run = Path(args.run)
        if (run / "history.csv").exists():
            training_figures(run)
        if (run / "predictions.csv").exists():
            evaluation_figures(run)
    else:
        from .engine import resolve_config, fit, evaluate
        if args.command in ("train", "adapt"):
            fit(resolve_config(args.config), args.output,
                getattr(args, "source_checkpoint", None), args.resume)
        elif args.command in ("evaluate", "transfer"):
            print(json.dumps(evaluate(args.checkpoint, args.data_dir, args.dataset,
                args.output, not args.no_trajectories, args.device), indent=2))
        elif args.command == "seeds":
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=False)
            if len(set(args.seeds)) != len(args.seeds):
                raise ValueError("Seeds must be distinct")
            summaries = []
            for seed in args.seeds:
                checkpoint = fit(resolve_config(args.config, {"seed": seed}), out / f"seed_{seed}")
                summaries.append({"seed": seed, "checkpoint": str(checkpoint)})
            (out / "seeds.json").write_text(json.dumps(summaries, indent=2))
        elif args.command == "smoke":
            from .synthetic import generate
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=False)
            generate(out / "data")
            c = resolve_config(overrides={"data_dir": str(out / "data"), "epochs": 2,
                "batch_size": 16, "window_size": 5, "device": "cpu", "validation_stride": 2,
                "model": {"hidden_size": 8, "num_layers": 1, "head_size": 4, "dropout": 0.}})
            checkpoint = fit(c, out / "train")
            result = evaluate(checkpoint, out / "data", "FD001", out / "evaluation", device="cpu")
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
