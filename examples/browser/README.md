# Browser Use Examples

This directory contains example scripts for browser automation using different configurations.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your configuration:
   - Copy `config.sample.json` to `config.json`
   - Edit `config.json` and add your Gemini API key

## Available Scripts

### configurable_browser.py
A flexible script that allows you to run browser automation tasks with customizable settings:

```bash
# Basic usage with direct task input
python configurable_browser.py --task "Your task description here"

# Using a task file (useful for longer tasks)
python configurable_browser.py --task-file "path/to/task.txt"

# With custom User-Agent
python configurable_browser.py --task "Your task" --user-agent "Custom User Agent String"

# Run in headless mode
python configurable_browser.py --task "Your task" --headless

# Use custom config file location
python configurable_browser.py --task "Your task" --config "/path/to/config.json"
```

A sample task file (`sample_task.txt`) is provided as an example of how to structure your task descriptions.

### gradio_demo.py
A web UI for running browser automation tasks. Supports multiple LLM models:

```bash
python gradio_demo.py
```

Note: Make sure you have set up your API keys properly before running the tasks.