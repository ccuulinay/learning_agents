# Create adk agent
```bash
adk create my_agent
```

# Run with command-line interface¶
Run your agent using the adk run command-line tool.
```bash
adk run my_agent
```

# Run with web interface
The ADK framework provides web interface you can use to test and interact with your agent. You can start the web interface using the following command:

```bash
adk web --port 8038

adk web --port 8038 --log_level DEBUG
```


>Note:  
>Run this command from the parent directory that contains your my_agent/ folder. For example, if your agent is inside agents/my_agent/, run adk web from the agents/ directory.


# Eval with custom evalset and config
```bash
adk eval crs_home_automation_agent crs_home_automation_agent/integration.evalset.json --config_file_path=crs_home_automation_agent/test_config.json --print_detailed_results
```