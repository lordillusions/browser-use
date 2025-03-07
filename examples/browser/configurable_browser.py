import os
import json
import asyncio
import argparse
from typing import Optional
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig

# Default settings
DEFAULT_USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'

def load_config(config_path: Optional[str] = None) -> str:
    """Load API key from config file"""
    if not config_path:
        # Look for config.json in the same directory as this script
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    
    if not os.path.exists(config_path):
        sample_path = os.path.join(os.path.dirname(__file__), 'config.sample.json')
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            f"Please copy {sample_path} to {config_path} and add your API key."
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    if 'gemini_api_key' not in config:
        raise KeyError("gemini_api_key not found in config file")
        
    return config['gemini_api_key']

def read_task_from_file(file_path: str) -> str:
    """Read task description from a file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Task file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return f.read().strip()

async def run_browser_task(
    task: str,
    user_agent: Optional[str] = None,
    headless: bool = False,
    config_path: Optional[str] = None
) -> str:
    """
    Run a browser automation task with the specified configuration.
    
    Args:
        task: The task description for the agent to perform
        user_agent: Custom User-Agent string (optional)
        headless: Whether to run in headless mode (default: False)
        config_path: Path to config file containing API key (optional)
    """
    try:
        # Load API key from config
        api_key = load_config(config_path)

        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash-exp',
            api_key=SecretStr(api_key)
        )

        # Create browser context configuration
        context_config = BrowserContextConfig(
            user_agent=user_agent or DEFAULT_USER_AGENT
        )

        # Configure and initialize the browser
        browser = Browser(
            config=BrowserConfig(
                headless=headless,
                extra_chromium_args=[
                    '--no-first-run',
                    '--no-default-browser-check'
                ],
                new_context_config=context_config
            )
        )
        
        # Create and run the agent
        agent = Agent(
            task=task,
            llm=llm,
            browser=browser
        )
        
        result = await agent.run()
        await browser.close()
        return result

    except Exception as e:
        return f'Error: {str(e)}'

def main():
    parser = argparse.ArgumentParser(description='Run browser automation tasks with custom configuration')
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument('--task', help='The task description')
    task_group.add_argument('--task-file', help='Path to file containing the task description')
    parser.add_argument('--user-agent', help='Custom User-Agent string')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--config', help='Path to config file containing API key')
    
    args = parser.parse_args()
    
    # Get task from either direct input or file
    task = args.task if args.task else read_task_from_file(args.task_file)
    
    result = asyncio.run(run_browser_task(
        task=task,
        user_agent=args.user_agent,
        headless=args.headless,
        config_path=args.config
    ))
    
    print(result)

if __name__ == '__main__':
    main()