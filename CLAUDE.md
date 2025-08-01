# DRAEM Project Guidelines

## Core Requirements

### Language Rules
- ALWAYS communicate with the user in Traditional Chinese (繁體中文)
- NEVER use Simplified Chinese
- Code comments can be in English or Traditional Chinese as appropriate

### Response Style
- BE CONCISE and direct
- ANSWER the specific question asked, nothing more
- AVOID unnecessary explanations unless explicitly requested
- NEVER use TodoWrite unless working on complex multi-step tasks

### Task Management
- PRIORITIZE answering the user's immediate question first
- DO NOT get stuck on previous tasks or todos
- WHEN user asks a new question, respond to it directly before returning to previous work
- ONLY return to previous tasks after explicit confirmation from the user

### Testing Workflow
- COMPLETE one task and show results before proceeding to the next
- WAIT for user confirmation after presenting results
- DO NOT rush to implement multiple steps without verification
- PRESENT test results clearly and ask for approval before continuing

## Code Style Rules

### General Principles
- NEVER add comments unless explicitly requested by the user
- MAINTAIN existing code style and conventions
- PRESERVE original indentation and formatting
- DO NOT refactor working code unless asked

### Python Specific
- FOLLOW PEP 8 only where the existing code already does
- KEEP existing import order and style
- PRESERVE existing variable naming conventions
- MAINTAIN existing function/class structure

### Modifications
- MAKE minimal changes to achieve the requested goal
- ALWAYS explain what changes you're making and why
- TEST your understanding by asking clarifying questions when needed
- NEVER assume implementation details - ask for clarification

## Dataset-Specific Guidelines

This project contains processing workflows for multiple datasets. Related documentation:
- Optical dataset: CLAUDE_optical.md
- [Future datasets]: CLAUDE_[dataset].md

When discussing a specific dataset, please explicitly mention which documentation file to reference.