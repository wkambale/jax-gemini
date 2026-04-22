# Architecture
The library uses `google-generativeai` to solicit Python code using Gemini Pro. We validate AST with an import whitelist before passing it into an isolated namespace where it executes.
