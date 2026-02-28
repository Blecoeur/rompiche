"""
TUI Dashboard for Rompiche Optimization
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal
    from textual.widgets import Static, ProgressBar, DataTable, Label
    from textual.reactive import reactive
    TUI_AVAILABLE = True
except ImportError:
    TUI_AVAILABLE = False


class MetricsTracker:
    """Tracks all metrics and history for the TUI dashboard"""
    
    def __init__(self):
        self.iteration_metrics = []
        self.current_iteration = 0
        self.total_iterations = 0
        self.start_time = datetime.now()
        self.tokens_used = 0
        self.current_status = "Initializing..."
        self.current_progress = 0
        self.total_items = 0
        self.mismatch_examples = []
        self.paused = False
        self.stopped = False
    
    def update_status(self, status: str):
        """Update the current status message"""
        self.current_status = status
    
    def update_progress(self, processed: int, total: int):
        """Update processing progress"""
        self.current_progress = processed
        self.total_items = total
    
    def add_iteration_metrics(self, metrics: dict):
        """Add metrics for the current iteration"""
        self.iteration_metrics.append(metrics)
        self.current_iteration += 1
    
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
        elapsed = datetime.now() - self.start_time
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
        self.update_status("STOPPED by user")


class LiveDashboard(App):
    """Main TUI Dashboard Application"""
    
    CSS = """
    #main-container {
        layout: vertical;
    }

    .header {
        height: 3;
        background: $primary;
        color: white;
        text-align: center;
    }

    .section-title {
        background: $secondary;
        color: white;
        padding: 0 1;
        height: 1;
    }

    .status-bar {
        height: 3;
        background: $surface;
        padding: 0 2;
    }

    .side-by-side {
        height: auto;
    }

    .metrics-section {
        width: 1fr;
        height: 10;
    }

    .performance-section {
        width: 1fr;
        height: 10;
    }

    .progress-section {
        width: 1fr;
        height: 6;
    }

    .mismatches-section {
        width: 1fr;
        height: 10;
    }

    .controls {
        dock: bottom;
        height: 1;
        background: $surface;
        text-align: center;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("p", "pause_resume", "Pause/Resume"),
        ("s", "stop", "Stop"),
    ]
    
    def __init__(self, tracker: MetricsTracker):
        super().__init__()
        self.tracker = tracker
        self.refresh_interval = 0.1  # seconds
        self._columns_initialized = False
    
    def compose(self) -> ComposeResult:
        yield Container(
            Label("ROMPICHE OPTIMIZATION DASHBOARD", classes="header"),
            Container(
                Label("🎯 CURRENT ITERATION: 0/0", id="iteration-info"),
                Label("⏱️  ELAPSED TIME: 00:00:00", id="time-info"),
                Label("💾 TOKENS USED: 0", id="tokens-info"),
                classes="status-bar"
            ),
            Horizontal(
                Container(
                    Label("📊 OVERALL METRICS", classes="section-title"),
                    DataTable(id="metrics-table"),
                    classes="metrics-section"
                ),
                Container(
                    Label("📈 PERFORMANCE EVOLUTION", classes="section-title"),
                    ProgressBar(id="performance-progress", total=100),
                    Static(id="performance-text"),
                    classes="performance-section"
                ),
                classes="side-by-side"
            ),
            Horizontal(
                Container(
                    Label("🔄 CURRENT ITERATION DETAILS", classes="section-title"),
                    ProgressBar(id="iteration-progress", total=100),
                    Label("Processing: 0/0 items (0%)", id="progress-text"),
                    Label("Status: Initializing...", id="status-text"),
                    classes="progress-section"
                ),
                Container(
                    Label("💡 RECENT MISMATCHES", classes="section-title"),
                    Static(id="mismatch-content"),
                    classes="mismatches-section"
                ),
                classes="side-by-side"
            ),
            Label("🎛️  [Q] Quit  [P] Pause/Resume  [S] Stop", classes="controls"),
            id="main-container"
        )
    
    def on_mount(self) -> None:
        self.update_metrics_table()
        self.update_performance_chart()
        self.set_interval(self.refresh_interval, self.update_display)
    
    def update_display(self) -> None:
        """Update all dashboard components"""
        if self.tracker.stopped:
            self.exit()
            return
            
        self.update_status_bar()
        self.update_metrics_table()
        self.update_performance_chart()
        self.update_progress_section()
        self.update_mismatches()
    
    def update_status_bar(self) -> None:
        """Update the status bar information"""
        iteration_label = self.query_one("#iteration-info", Label)
        time_label = self.query_one("#time-info", Label)
        tokens_label = self.query_one("#tokens-info", Label)

        iteration_label.update(f"🎯 CURRENT ITERATION: {self.tracker.current_iteration}/{self.tracker.total_iterations}")
        time_label.update(f"⏱️  ELAPSED TIME: {self.tracker.get_elapsed_time()}")
        tokens_label.update(f"💾 TOKENS USED: {self.tracker.tokens_used:,}")
    
    def update_metrics_table(self) -> None:
        """Update the metrics data table"""
        table = self.query_one("#metrics-table", DataTable)
        overall_metrics = self.tracker.get_overall_metrics()

        all_metric_names = sorted({
            metric
            for field_metrics in overall_metrics.values()
            for metric in field_metrics
        })

        if not self._columns_initialized and all_metric_names:
            table.add_column("Field")
            for metric_name in all_metric_names:
                table.add_column(metric_name.replace("_", " ").title())
            self._columns_initialized = True

        table.clear()

        for field, metrics in overall_metrics.items():
            row = [field]
            for metric_name in all_metric_names:
                if metric_name in metrics:
                    row.append(f"{metrics[metric_name]:.2f}")
                else:
                    row.append("")
            table.add_row(*row)
    
    def update_performance_chart(self) -> None:
        """Update the performance evolution chart"""
        if not self.tracker.iteration_metrics:
            return

        performance_text = self.query_one("#performance-text", Static)
        performance_scores = []

        for i, metrics in enumerate(self.tracker.iteration_metrics):
            # Calculate composite score
            all_values = []
            for field_metrics in metrics.values():
                all_values.extend(field_metrics.values())
            score = sum(all_values) / len(all_values) if all_values else 0
            performance_scores.append(score)

        # Update progress bar and text
        progress_bar = self.query_one("#performance-progress", ProgressBar)
        current_score = performance_scores[-1] * 100
        progress_bar.update(total=100, progress=current_score)

        performance_lines = []
        for i, score in enumerate(performance_scores):
            bar_length = int(score * 50)
            performance_lines.append(f"Iteration {i+1}: {'█' * bar_length}{' ' * (50 - bar_length)} {score:.2%}")
        
        performance_text.update("\n".join(performance_lines))
    
    def update_progress_section(self) -> None:
        """Update the current iteration progress section"""
        progress_bar = self.query_one("#iteration-progress", ProgressBar)
        progress_text = self.query_one("#progress-text", Label)
        status_text = self.query_one("#status-text", Label)

        if self.tracker.total_items > 0:
            progress = min(self.tracker.current_progress / self.tracker.total_items, 1.0)
            progress_bar.update(total=self.tracker.total_items, progress=self.tracker.current_progress)
            progress_text.update(f"Processing: {self.tracker.current_progress}/{self.tracker.total_items} items ({progress:.0%})")
        else:
            progress_bar.update(total=100, progress=0)
            progress_text.update("Processing: 0/0 items (0%)")

        status = self.tracker.current_status
        if self.tracker.paused:
            status = "⏸️  " + status
        status_text.update(f"Status: {status}")
    
    def update_mismatches(self) -> None:
        """Update the recent mismatches display"""
        mismatch_content = self.query_one("#mismatch-content", Static)
        if not self.tracker.mismatch_examples:
            mismatch_content.update("No recent mismatches - wait for the first iteration to complete")
            return

        content = []
        for i, example in enumerate(self.tracker.mismatch_examples):
            content.append(f"Mismatch {i+1}:")
            content.append(f"  Input: {example.get('input', '')[:70]}...")
            content.append(f"  Expected: {json.dumps(example.get('ground_truth', {}))}")
            content.append(f"  Predicted: {json.dumps(example.get('prediction', {}))}")
            if i < len(self.tracker.mismatch_examples) - 1:
                content.append("─" * 70)

        mismatch_content.update("\n".join(content))
    
    def action_pause_resume(self) -> None:
        """Handle pause/resume action"""
        if self.tracker.paused:
            self.tracker.resume()
        else:
            self.tracker.pause()
    
    def action_stop(self) -> None:
        """Handle stop action"""
        self.tracker.stop()
    
    def action_quit(self) -> None:
        """Handle quit action"""
        self.exit()


def check_tui_available():
    """Check if TUI dependencies are available"""
    return TUI_AVAILABLE