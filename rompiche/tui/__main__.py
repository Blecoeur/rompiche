"""
TUI Dashboard Entry Point
"""

from rompiche.tui.dashboard import LiveDashboard, check_tui_available


def main():
    """Main entry point for the TUI dashboard"""
    if not check_tui_available():
        print("Error: TUI dependencies not available.")
        print("Please install them with: pip install rompiche[tui]")
        return 1

    # Create and run the dashboard
    dashboard = LiveDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
