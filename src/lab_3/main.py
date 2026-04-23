# from functools import partial
# import pandas as pd

# from src.llm import llm_call
# from src.baseline import run_baseline
# from src.agent import run_agent
# from src.experiment import load_topics, run_experiment, summarize_records
# from src.plotting import plot_main_comparison, plot_factor_experiment
# from src.utils import load_json, ensure_dir


# def main():
#     cfg = load_json("configs/experiment_config.json")
#     topics = load_topics(cfg["topics_path"])
#     output_dir = cfg["output_dir"]
#     ensure_dir(output_dir)

#     default_top_k = cfg["default_top_k"]
#     default_max_steps = cfg["default_max_steps"]
#     min_sources = cfg["min_sources"]

#     all_records = []

#     # Эксперимент 1. Baseline
#     baseline_runner = run_baseline
#     all_records += run_experiment(
#         topics=topics,
#         mode_name="baseline",
#         runner=baseline_runner,
#         llm_call=llm_call,
#         output_dir=output_dir
#     )

#     # Эксперимент 2. Agent without evaluator
#     agent_runner = partial(
#         run_agent,
#         max_steps=default_max_steps,
#         top_k=default_top_k,
#         min_sources=min_sources,
#         use_evaluator=False
#     )
#     all_records += run_experiment(
#         topics=topics,
#         mode_name="agent",
#         runner=agent_runner,
#         llm_call=llm_call,
#         output_dir=output_dir
#     )

#     # Эксперимент 3. Agent with evaluator
#     agent_eval_runner = partial(
#         run_agent,
#         max_steps=default_max_steps,
#         top_k=default_top_k,
#         min_sources=min_sources,
#         use_evaluator=True
#     )
#     all_records += run_experiment(
#         topics=topics,
#         mode_name="agent_evaluator",
#         runner=agent_eval_runner,
#         llm_call=llm_call,
#         output_dir=output_dir
#     )

#     df, summary = summarize_records(all_records, output_dir=output_dir)
#     print(summary)

#     plot_main_comparison(summary, output_dir=f"{output_dir}/figures")

#     # Эксперимент 4. top_k = 3, 5, 8
#     factor_records = []
#     for top_k in [3, 5, 8]:
#         runner = partial(
#             run_agent,
#             max_steps=default_max_steps,
#             top_k=top_k,
#             min_sources=min_sources,
#             use_evaluator=False
#         )
#         records = run_experiment(
#             topics=topics,
#             mode_name=f"agent_topk_{top_k}",
#             runner=runner,
#             llm_call=llm_call,
#             output_dir=output_dir
#         )
#         for r in records:
#             r["top_k"] = top_k
#         factor_records.extend(records)

#     df_topk = pd.DataFrame(factor_records)
#     df_topk.to_csv(f"{output_dir}/results/topk_experiment.csv", index=False, encoding="utf-8")
#     plot_factor_experiment(
#         df_topk,
#         factor_col="top_k",
#         metric_col="rubric",
#         output_path=f"{output_dir}/figures/topk_vs_rubric.png"
#     )
#     plot_factor_experiment(
#         df_topk,
#         factor_col="top_k",
#         metric_col="latency",
#         output_path=f"{output_dir}/figures/topk_vs_latency.png"
#     )

#     # Эксперимент 5. max_steps = 4, 6, 8
#     factor_records = []
#     for max_steps in [4, 6, 8]:
#         runner = partial(
#             run_agent,
#             max_steps=max_steps,
#             top_k=default_top_k,
#             min_sources=min_sources,
#             use_evaluator=False
#         )
#         records = run_experiment(
#             topics=topics,
#             mode_name=f"agent_steps_{max_steps}",
#             runner=runner,
#             llm_call=llm_call,
#             output_dir=output_dir
#         )
#         for r in records:
#             r["max_steps"] = max_steps
#         factor_records.extend(records)

#     df_steps = pd.DataFrame(factor_records)
#     df_steps.to_csv(f"{output_dir}/results/max_steps_experiment.csv", index=False, encoding="utf-8")
#     plot_factor_experiment(
#         df_steps,
#         factor_col="max_steps",
#         metric_col="rubric",
#         output_path=f"{output_dir}/figures/max_steps_vs_rubric.png"
#     )
#     plot_factor_experiment(
#         df_steps,
#         factor_col="max_steps",
#         metric_col="latency",
#         output_path=f"{output_dir}/figures/max_steps_vs_latency.png"
#     )


# if __name__ == "__main__":
#     main()

from functools import partial
import logging
import sys
import time

import pandas as pd

from src.llm import llm_call
from src.baseline import run_baseline
from src.agent import run_agent
from src.experiment import load_topics, run_experiment, summarize_records
from src.plotting import plot_main_comparison, plot_factor_experiment
from src.utils import load_json, ensure_dir


def setup_logging(output_dir: str) -> logging.Logger:
    log_dir = f"{output_dir}/logs"
    ensure_dir(log_dir)

    logger = logging.getLogger("lab3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(f"{log_dir}/run.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def run_and_log_experiment(logger, *, topics, mode_name, runner, llm_call, output_dir):
    logger.info("START experiment: %s | topics=%d", mode_name, len(topics))
    start = time.time()

    try:
        records = run_experiment(
            topics=topics,
            mode_name=mode_name,
            runner=runner,
            llm_call=llm_call,
            output_dir=output_dir
        )
        elapsed = time.time() - start
        logger.info(
            "DONE experiment: %s | records=%d | elapsed=%.2f sec",
            mode_name, len(records), elapsed
        )
        return records
    except Exception:
        logger.exception("FAILED experiment: %s", mode_name)
        raise


def main():
    cfg = load_json("configs/experiment_config.json")
    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)

    logger = setup_logging(output_dir)
    logger.info("Application started")
    logger.info("Loading experiment config from configs/experiment_config.json")

    topics = load_topics(cfg["topics_path"])
    logger.info("Loaded topics from %s | count=%d", cfg["topics_path"], len(topics))

    default_top_k = cfg["default_top_k"]
    default_max_steps = cfg["default_max_steps"]
    min_sources = cfg["min_sources"]

    logger.info(
        "Experiment params | default_top_k=%s | default_max_steps=%s | min_sources=%s | output_dir=%s",
        default_top_k, default_max_steps, min_sources, output_dir
    )

    all_records = []

    # Эксперимент 1. Baseline
    baseline_runner = run_baseline
    all_records += run_and_log_experiment(
        logger,
        topics=topics,
        mode_name="baseline",
        runner=baseline_runner,
        llm_call=llm_call,
        output_dir=output_dir
    )

    # Эксперимент 2. Agent without evaluator
    agent_runner = partial(
        run_agent,
        max_steps=default_max_steps,
        top_k=default_top_k,
        min_sources=min_sources,
        use_evaluator=False
    )
    all_records += run_and_log_experiment(
        logger,
        topics=topics,
        mode_name="agent",
        runner=agent_runner,
        llm_call=llm_call,
        output_dir=output_dir
    )

    # Эксперимент 3. Agent with evaluator
    agent_eval_runner = partial(
        run_agent,
        max_steps=default_max_steps,
        top_k=default_top_k,
        min_sources=min_sources,
        use_evaluator=True
    )
    all_records += run_and_log_experiment(
        logger,
        topics=topics,
        mode_name="agent_evaluator",
        runner=agent_eval_runner,
        llm_call=llm_call,
        output_dir=output_dir
    )

    logger.info("Summarizing all records | total=%d", len(all_records))
    df, summary = summarize_records(all_records, output_dir=output_dir)

    logger.info("Summary created")
    logger.info("\n%s", summary)
    print(summary)

    logger.info("Plotting main comparison")
    plot_main_comparison(summary, output_dir=f"{output_dir}/figures")

    # Эксперимент 4. top_k = 3, 5, 8
    factor_records = []
    for top_k in [3, 5, 8]:
        logger.info("Factor experiment: top_k=%s", top_k)

        runner = partial(
            run_agent,
            max_steps=default_max_steps,
            top_k=top_k,
            min_sources=min_sources,
            use_evaluator=False
        )
        records = run_and_log_experiment(
            logger,
            topics=topics,
            mode_name=f"agent_topk_{top_k}",
            runner=runner,
            llm_call=llm_call,
            output_dir=output_dir
        )
        for r in records:
            r["top_k"] = top_k
        factor_records.extend(records)

    df_topk = pd.DataFrame(factor_records)
    topk_csv = f"{output_dir}/results/topk_experiment.csv"
    logger.info("Saving top_k experiment csv to %s | rows=%d", topk_csv, len(df_topk))
    df_topk.to_csv(topk_csv, index=False, encoding="utf-8")

    logger.info("Plotting top_k vs rubric")
    plot_factor_experiment(
        df_topk,
        factor_col="top_k",
        metric_col="rubric",
        output_path=f"{output_dir}/figures/topk_vs_rubric.png"
    )

    logger.info("Plotting top_k vs latency")
    plot_factor_experiment(
        df_topk,
        factor_col="top_k",
        metric_col="latency",
        output_path=f"{output_dir}/figures/topk_vs_latency.png"
    )

    # Эксперимент 5. max_steps = 4, 6, 8
    factor_records = []
    for max_steps in [4, 6, 8]:
        logger.info("Factor experiment: max_steps=%s", max_steps)

        runner = partial(
            run_agent,
            max_steps=max_steps,
            top_k=default_top_k,
            min_sources=min_sources,
            use_evaluator=False
        )
        records = run_and_log_experiment(
            logger,
            topics=topics,
            mode_name=f"agent_steps_{max_steps}",
            runner=runner,
            llm_call=llm_call,
            output_dir=output_dir
        )
        for r in records:
            r["max_steps"] = max_steps
        factor_records.extend(records)

    df_steps = pd.DataFrame(factor_records)
    steps_csv = f"{output_dir}/results/max_steps_experiment.csv"
    logger.info("Saving max_steps experiment csv to %s | rows=%d", steps_csv, len(df_steps))
    df_steps.to_csv(steps_csv, index=False, encoding="utf-8")

    logger.info("Plotting max_steps vs rubric")
    plot_factor_experiment(
        df_steps,
        factor_col="max_steps",
        metric_col="rubric",
        output_path=f"{output_dir}/figures/max_steps_vs_rubric.png"
    )

    logger.info("Plotting max_steps vs latency")
    plot_factor_experiment(
        df_steps,
        factor_col="max_steps",
        metric_col="latency",
        output_path=f"{output_dir}/figures/max_steps_vs_latency.png"
    )

    logger.info("Application finished successfully")


if __name__ == "__main__":
    main()