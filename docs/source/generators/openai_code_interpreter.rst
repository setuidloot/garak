garak.generators.openai_code_interpreter
========================================

This inactive generator uses the OpenAI Responses API with exactly one built-in
tool: Code Interpreter. The target receives no instruction to use the tool;
tool selection remains part of the evaluated target's behaviour. Each
independent conversation branch receives an explicit container, which is
retained across that branch's turns and deleted when the generator closes.

The generator requires ``OPENAI_API_KEY`` and a Responses target that supports
Code Interpreter::

   python -m garak --target_type openai_code_interpreter \
       --target_name <target> --probes <probe>

.. automodule:: garak.generators.openai_code_interpreter
   :members:
   :undoc-members:
   :show-inheritance:
