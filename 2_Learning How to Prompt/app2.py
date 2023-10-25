## Bad Prompt Practices

# Now, let’s see some examples of prompting that are generally considered bad.

# Here’s an example of a too-vague prompt that provides little context or guidance for the model to generate a meaningful response.

from langchain import PromptTemplate

template = "Tell me something about {topic}."
prompt = PromptTemplate(
    input_variables=["topic"],
    template=template,
)
prompt.format(topic="dogs")


