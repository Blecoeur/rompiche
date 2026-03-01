from datetime import datetime

class MetricsTracker:
    """Tracks all metrics and history for the TUI dashboard"""

    def __init__(self):
        self.iteration_metrics = []  # test-set metrics only (used by dashboard)
        self.train_iteration_metrics = []
        self.current_iteration = 0
        self.total_iterations = 0
        self.start_time = datetime.now()
        self.stop_time = None
        self.tokens_used = 0
        self.current_status = "Initializing..."
        self.current_progress = 0
        self.total_items = 0
        self.mismatch_examples = []
        self.paused = False
        self.stopped = False
        self.current_prompt = ""
        self.current_schema = {}
        self.update_history = []
        self.last_update_at = None
        self.user_hints = []
        self.evaluator = None  # Store evaluator for mismatch analysis

    def update_status(self, status: str):
        """Update the current status message"""
        self.current_status = status

    def update_progress(self, processed: int, total: int):
        """Update processing progress"""
        self.current_progress = processed
        self.total_items = total

    def add_iteration_metrics(self, metrics: dict, dataset_type: str = "test"):
        """Add metrics for the current iteration"""
        if dataset_type == "test":
            self.iteration_metrics.append(metrics)
        else:
            self.train_iteration_metrics.append(metrics)

    def add_tokens(self, tokens: int):
        """Add to the total token count"""
        self.tokens_used += tokens

    def add_mismatch(self, example: dict):
        """Add a mismatch example (keep only 10 most recent)"""
        if len(self.mismatch_examples) < 10:
            self.mismatch_examples.append(example)
        else:
            self.mismatch_examples.pop(0)
            self.mismatch_examples.append(example)

    def get_elapsed_time(self) -> str:
        """Get formatted elapsed time"""
        end_time = (
            self.stop_time
            if hasattr(self, "stop_time") and self.stop_time
            else datetime.now()
        )
        elapsed = end_time - self.start_time
        hours, remainder = divmod(elapsed.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_overall_metrics(self) -> dict:
        """Calculate average metrics across all iterations"""
        if not self.iteration_metrics:
            return {}

        overall = {}
        for field in self.iteration_metrics[0].keys():
            overall[field] = {}
            for metric in self.iteration_metrics[0][field].keys():
                values = [m[field][metric] for m in self.iteration_metrics]
                overall[field][metric] = sum(values) / len(values)
        return overall

    def get_current_iteration_metrics(self) -> dict:
        """Get metrics for the current iteration"""
        if self.iteration_metrics:
            return self.iteration_metrics[-1]
        return {}

    def pause(self):
        """Pause the optimization"""
        self.paused = True
        self.update_status("PAUSED - Press P to continue")

    def resume(self):
        """Resume the optimization"""
        self.paused = False
        self.update_status("Resumed")

    def stop(self):
        """Stop the optimization"""
        self.stopped = True
        if self.stop_time is None:
            self.stop_time = datetime.now()
        self.update_status("STOPPED by user")

    def freeze_elapsed_time(self):
        """Freeze elapsed time display at the current moment."""
        if self.stop_time is None:
            self.stop_time = datetime.now()

    def set_active_configuration(self, prompt: str, schema: dict):
        """Store the currently active prompt and schema."""
        self.current_prompt = prompt or ""
        self.current_schema = schema or {}

    def add_brain_update(self, update: dict):
        """Append a brain update entry and stamp the latest update time."""
        self.last_update_at = datetime.now()
        stamped_update = {
            "timestamp": self.last_update_at.strftime("%H:%M:%S"),
            **update,
        }
        self.update_history.append(stamped_update)
        if len(self.update_history) > 50:
            self.update_history = self.update_history[-50:]