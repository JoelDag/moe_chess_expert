try:
    import wandb
except ImportError:  # pragma: no cover - optional logging backend
    wandb = None

from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from transformers import Trainer

def evaluate_on_harness(model, tasks_list):
    """
    Evaluates a model on a list of lm-harness tasks.
    """
    
    task_manager = tasks.TaskManager()

    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks_list,
        task_manager=task_manager,
        limit=10,
        num_fewshot=0
    )

    return results

class HarnessTrainer(Trainer):
    def __init__(self, *args, harness_tasks=None, do_harness_eval=False, wandb_project="chess_expert_moe", **kwargs):
        self.harness_tasks = harness_tasks or []
        self.do_harness_eval = do_harness_eval
        self.wandb_project = wandb_project
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # TODO: do we want to run default eval? i think not necessary
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Run lm-harness evaluation
        if self.do_harness_eval:
            print("Running lm-harness evaluation...")
            
            hflm = HFLM(pretrained=self.model, tokenizer=self.tokenizer)
            results = evaluate_on_harness(hflm, self.harness_tasks)
            
            # Log the results
            for task_name, task_results in results['results'].items():
                for metric, value in task_results.items():
                    if "stderr" not in metric:
                        metrics[f"{metric_key_prefix}_harness_{task_name}_{metric}"] = value
            
            self.log(metrics)
            if wandb is not None:
                if wandb.run is None:
                    wandb.init(project=self.wandb_project)
                wandb.log(metrics)

        return metrics
