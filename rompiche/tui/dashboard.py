"""
TUI Dashboard for Rompiche Optimization
"""

import json
from datetime import datetime

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, VerticalScroll
    from textual.widgets import (
        Static,
        ProgressBar,
        DataTable,
        Label,
        Input,
        Footer,
        Sparkline,
    )
    from textual.screen import ModalScreen
    from textual import events

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
        self.update_status("STOPPED by user")

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
        height: 20;
        layout: vertical;
        overflow: hidden;
    }

    #sparkline-container {
        height: 1fr;
        overflow-y: auto;
        padding: 0 1;
    }

    .sparkline-group {
        height: auto;
        padding: 0 1;
    }

    .sparkline-label {
        height: 1;
    }

    Sparkline {
        height: 3;
        margin: 0 1;
    }

    .sparkline-value {
        height: 1;
        text-align: center;
    }

    .progress-section {
        width: 1fr;
        height: 6;
    }

    .mismatches-section {
        width: 1fr;
        height: 20;
        layout: vertical;
        overflow: hidden;
    }

    #mismatch-scroll {
        height: 1fr;
        overflow-y: auto;
    }

    #mismatch-content {
        height: auto;
        padding: 0 1;
    }

    .decision-section {
        height: 20;
    }

    .prompt-section {
        width: 1fr;
        height: 1fr;
        layout: vertical;
    }

    .updates-section {
        width: 1fr;
        height: 1fr;
        layout: vertical;
    }

    #prompt-schema-scroll {
        height: 1fr;
        overflow-y: auto;
    }

    #updates-scroll {
        height: 1fr;
        overflow-y: auto;
    }

    #prompt-schema-content {
        height: auto;
        overflow-y: auto;
        padding: 0 1;
    }

    #updates-content {
        height: auto;
        overflow-y: auto;
        padding: 0 1;
    }

    .row-spacer {
        height: 1;
    }


    Footer {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("p", "pause_resume", "Pause/Resume"),
        ("s", "stop", "Stop"),
        ("h", "add_hint", "Add Hint"),
    ]

    def __init__(self, tracker: MetricsTracker):
        super().__init__()
        self.tracker = tracker
        self.refresh_interval = 0.1  # seconds
        self._columns_initialized = False
        self._last_prompt_schema_text = ""
        self._last_updates_text = ""

    def compose(self) -> ComposeResult:
        yield Container(
            Label("💤 ROMPICHE OPTIMIZATION DASHBOARD", classes="header"),
            Container(
                Label("🎯 CURRENT ITERATION: 0/0", id="iteration-info"),
                Label("⏱️  ELAPSED TIME: 00:00:00", id="time-info"),
                Label("💾 TOKENS USED: 0", id="tokens-info"),
                classes="status-bar",
            ),
            Horizontal(
                Container(
                    Label("🔄 CURRENT ITERATION DETAILS", classes="section-title"),
                    ProgressBar(id="iteration-progress", total=100),
                    Label("Processing: 0/0 items (0%)", id="progress-text"),
                    Label("Status: Initializing...", id="status-text"),
                    classes="progress-section",
                ),
                Container(
                    Label("📊 CURRENT METRICS", classes="section-title"),
                    DataTable(id="metrics-table"),
                    classes="metrics-section",
                ),
                classes="side-by-side",
            ),
            Horizontal(
                Container(
                    Label("📈 PERFORMANCE EVOLUTION", classes="section-title"),
                    VerticalScroll(id="sparkline-container"),
                    classes="performance-section",
                ),
                Container(
                    Label("💡 RECENT MISMATCHES", classes="section-title"),
                    VerticalScroll(Static(id="mismatch-content"), id="mismatch-scroll"),
                    classes="mismatches-section",
                ),
                classes="side-by-side",
            ),
            Container(classes="row-spacer"),  # Spacer between 2nd and 3rd row
            Horizontal(
                Container(
                    Label("📝 CURRENT PROMPT & SCHEMA", classes="section-title"),
                    VerticalScroll(
                        Static(id="prompt-schema-content"), id="prompt-schema-scroll"
                    ),
                    classes="prompt-section",
                ),
                Container(
                    Label("🤖 ALL UPDATES", classes="section-title"),
                    VerticalScroll(Static(id="updates-content"), id="updates-scroll"),
                    classes="updates-section",
                ),
                classes="decision-section",
            ),
            Footer(),
            id="main-container",
        )

    def on_mount(self) -> None:
        self.update_metrics_table()
        self.update_performance_chart()
        self.update_prompt_schema()
        self.update_all_updates()
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
        self.update_prompt_schema()
        self.update_all_updates()

    def update_status_bar(self) -> None:
        """Update the status bar information"""
        iteration_label = self.query_one("#iteration-info", Label)
        time_label = self.query_one("#time-info", Label)
        tokens_label = self.query_one("#tokens-info", Label)

        iteration_label.update(
            f"🎯 CURRENT ITERATION: {self.tracker.current_iteration}/{self.tracker.total_iterations}"
        )
        time_label.update(f"⏱️  ELAPSED TIME: {self.tracker.get_elapsed_time()}")
        tokens_label.update(f"💾 TOKENS USED: {self.tracker.tokens_used:,}")

    def update_metrics_table(self) -> None:
        """Update the metrics data table with current iteration metrics"""
        table = self.query_one("#metrics-table", DataTable)
        current_metrics = self.tracker.get_current_iteration_metrics()

        # Get all metric names from current metrics
        all_metric_names = (
            sorted(
                {
                    metric
                    for field_metrics in current_metrics.values()
                    for metric in field_metrics
                }
            )
            if current_metrics
            else []
        )

        if not self._columns_initialized and all_metric_names:
            table.add_column("Field")
            for metric_name in all_metric_names:
                table.add_column(metric_name.replace("_", " ").title())
            self._columns_initialized = True

        table.clear()

        # If no current metrics available, show empty table
        if not current_metrics:
            return

        for field, metrics in current_metrics.items():
            row = [field]
            for metric_name in all_metric_names:
                if metric_name in metrics:
                    row.append(f"{metrics[metric_name]:.2f}")
                else:
                    row.append("")
            table.add_row(*row)

    def update_performance_chart(self) -> None:
        """Update the performance evolution sparklines per field-metric"""
        if not self.tracker.iteration_metrics:
            return

        field_metric_history: dict[tuple[str, str], list[float]] = {}
        for metrics in self.tracker.iteration_metrics:
            for field, field_metrics in metrics.items():
                for metric_name, value in field_metrics.items():
                    key = (field, metric_name)
                    if key not in field_metric_history:
                        field_metric_history[key] = []
                    field_metric_history[key].append(value)

        if not field_metric_history:
            return

        container = self.query_one("#sparkline-container", VerticalScroll)

        for (field, metric_name), values in field_metric_history.items():
            widget_id = f"spark-{field}-{metric_name}".replace(" ", "-")
            label_text = f"{field} - {metric_name.replace('_', ' ').title()}"
            current_text = f"Current: {values[-1]:.1%}"

            try:
                sparkline = self.query_one(f"#{widget_id}", Sparkline)
                sparkline.data = values
                value_label = self.query_one(f"#{widget_id}-val", Label)
                value_label.update(current_text)
            except Exception:
                group = Container(
                    Label(label_text, classes="sparkline-label"),
                    Sparkline(values, id=widget_id),
                    Label(
                        current_text, id=f"{widget_id}-val", classes="sparkline-value"
                    ),
                    classes="sparkline-group",
                )
                container.mount(group)

    def update_progress_section(self) -> None:
        """Update the current iteration progress section"""
        progress_bar = self.query_one("#iteration-progress", ProgressBar)
        progress_text = self.query_one("#progress-text", Label)
        status_text = self.query_one("#status-text", Label)

        if self.tracker.total_items > 0:
            progress = min(
                self.tracker.current_progress / self.tracker.total_items, 1.0
            )
            progress_bar.update(
                total=self.tracker.total_items, progress=self.tracker.current_progress
            )
            progress_text.update(
                f"Processing: {self.tracker.current_progress}/{self.tracker.total_items} items ({progress:.0%})"
            )
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
            mismatch_content.update(
                "No recent mismatches - wait for the first iteration to complete"
            )
            return

        content = []
        for i, example in enumerate(self.tracker.mismatch_examples):
            content.append(f"Mismatch {i + 1}:")
            content.append(f"  Input: {example.get('input', '')[:70]}...")

            # Get the evaluator to check which fields don't meet criteria
            evaluator = self.tracker.evaluator
            if evaluator and "ground_truth" in example and "prediction" in example:
                evaluation = evaluator.evaluate(
                    example["prediction"], example["ground_truth"]
                )
                success = evaluator.is_success(evaluation)

                if not success:
                    # Show only fields that don't meet criteria
                    problematic_fields = []
                    for field_name, field_metrics in evaluation.items():
                        for metric_name, score in field_metrics.items():
                            threshold = evaluator.success_thresholds.get(
                                field_name, {}
                            ).get(
                                metric_name,
                                1.0 if metric_name == "exact_match" else 0.8,
                            )
                            if score < threshold:
                                problematic_fields.append(field_name)
                                break

                    if problematic_fields:
                        content.append("  Problematic fields:")
                        for field in problematic_fields:
                            content.append(
                                f"    {field}: {json.dumps(example.get('ground_truth', {}).get(field, 'N/A'))} → {json.dumps(example.get('prediction', {}).get(field, 'N/A'))}"
                            )
                    else:
                        content.append(
                            f"  Expected: {json.dumps(example.get('ground_truth', {}))}"
                        )
                        content.append(
                            f"  Predicted: {json.dumps(example.get('prediction', {}))}"
                        )
                else:
                    content.append(
                        f"  Expected: {json.dumps(example.get('ground_truth', {}))}"
                    )
                    content.append(
                        f"  Predicted: {json.dumps(example.get('prediction', {}))}"
                    )
            else:
                content.append(
                    f"  Expected: {json.dumps(example.get('ground_truth', {}))}"
                )
                content.append(
                    f"  Predicted: {json.dumps(example.get('prediction', {}))}"
                )

            if i < len(self.tracker.mismatch_examples) - 1:
                content.append("─" * 70)

        mismatch_content.update("\n".join(content))

    def update_prompt_schema(self) -> None:
        """Update the current prompt and schema display"""
        try:
            prompt_content = self.query_one("#prompt-schema-content", Static)

            content = ""
            if self.tracker.current_prompt:
                content += f"Prompt:\n{self.tracker.current_prompt}\n\n"
            else:
                content += "Prompt: (initial prompt will appear here when first iteration starts)\n\n"

            # Always show the current schema, even if empty (will show as placeholder)
            if self.tracker.current_schema:
                schema_json = json.dumps(self.tracker.current_schema, indent=2)
                content += f"Schema:\n{schema_json}\n\n"
            else:
                content += "Schema: (initial schema will appear here when first iteration starts)\n\n"

            if self.tracker.last_update_at:
                content += (
                    f"Last update: {self.tracker.last_update_at.strftime('%H:%M:%S')}"
                )
            else:
                content += "Last update: waiting for first brain decision"

            if content != self._last_prompt_schema_text:
                prompt_content.update(content)
                self._last_prompt_schema_text = content
        except Exception as e:
            print(f"Error updating prompt schema: {e}")

    def update_all_updates(self) -> None:
        """Update the all updates display"""
        try:
            updates_content = self.query_one("#updates-content", Static)

            if not self.tracker.update_history:
                content = "No updates yet - waiting for first brain decision."
                if content != self._last_updates_text:
                    updates_content.update(content)
                    self._last_updates_text = content
                return

            content = ""
            for i, update in enumerate(self.tracker.update_history, start=1):
                iteration = update.get("iteration", "?")
                decision = update.get("decision", "continue")
                summary = update.get("summary", "No summary provided.")
                timestamp = update.get("timestamp", "--:--:--")
                content += f"{i}. [{timestamp}] Iteration {iteration} -> {decision}\n"
                content += f"   {summary}\n"

            if content != self._last_updates_text:
                updates_content.update(content)
                self._last_updates_text = content
        except Exception as e:
            print(f"Error updating all updates: {e}")

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

    def action_add_hint(self) -> None:
        """Handle add hint action - pauses processing while user provides input"""

        def on_submit(hint_text: str) -> None:
            if hint_text.strip():
                self.tracker.user_hints.append(hint_text.strip())
                self.tracker.update_status(f"Hint added: {hint_text[:50]}...")
            # Resume processing after hint is submitted
            if self.tracker.paused:
                self.tracker.resume()

        # Pause processing while user provides input
        if not self.tracker.paused:
            self.tracker.pause()

        self.push_screen(
            InputScreen("Add Hint", "Enter your hint for the LLM:", on_submit)
        )


class InputScreen(ModalScreen[str]):
    """A simple input screen for collecting user hints"""

    def __init__(self, title: str, prompt: str, callback: callable):
        super().__init__()
        self.title = title
        self.prompt = prompt
        self.callback = callback

    def compose(self) -> ComposeResult:
        yield Label(self.title, classes="header")
        yield Label(self.prompt)
        yield Input(placeholder="Type your hint here...", id="hint-input")
        yield Label("[Enter] Submit  [Esc] Cancel", classes="controls")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission"""
        self.callback(event.value)
        self.dismiss(event.value)

    def on_key(self, event: events.Key) -> None:
        """Handle key presses"""
        if event.key == "escape":
            self.dismiss("")


def check_tui_available():
    """Check if TUI dependencies are available"""
    return TUI_AVAILABLE
