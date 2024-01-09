""" 
This module contains the AIAssistant class, which represents an OpenAI Assistant.
"""

import json
import random
import time
from typing import List
from yaspin import yaspin
from openai import OpenAI
from openai import Client
from openai.types.beta import Thread, Assistant
from openai.types.beta.threads import Run, ThreadMessage
from openai_function_helper import Function, FunctionCall


PRINT_COLORS = [
    '\033[31m',
    '\033[32m',
    '\033[33m',
    '\033[34m',
    '\033[35m',
    '\033[36m',
]

class Message:
    """
    Represents a message in the conversation.
    """
    thread_id: str
    role: str
    content: str
    file_ids: List[str]

    def __init__(
        self, thread_id: str, role: str, content: str, file_ids: List[str] = None
    ):
        """
        Initializes a new instance of the class.

        Args:
            thread_id (str): The ID of the thread.
            role (str): The role of the user.
            content (str): The content of the message.
            file_ids (List[str], optional): The IDs of the files attached to the message. Defaults to None.
        """
        self.thread_id = thread_id
        self.role = role
        self.content = content
        self.file_ids = file_ids


class Conversation:
    """
    Represents a conversation.
    """
    messages: List[Message]

    def __init__(self, messages: List[Message]):
        self.messages = messages

    def print_conversation(self):
        """
        Prints the conversation.
        """
        for message in self.messages:
            print(f"{message.role}: {message.content}")
        for message in self.messages:
            print(f"{message.role}: {message.content}")


class AIAssistant:
    """
    Represents an AI Assistant.
    """
    assistant: Assistant
    client: OpenAI
    assistant_name: str
    assistant_description: str
    instruction: str
    model: str
    use_retrieval: bool
    use_code_interpreter: bool
    functions: List[Function]
    threads: List[Thread]
    tools: List[dict]
    file_ids: List[str]
    conversation: Conversation
    verbose: bool
    auto_delete: bool = True

    def __init__(
        self,
        instruction: str,
        model: str,
        use_retrieval: bool = False,
        use_code_interpreter: bool = False,
        file_ids: List[str] = None,
        functions: List[Function] = None,
        assistant_name: str = "AI Assistant",
        assistant_description: str = "An AI Assistant",
        verbose: bool = False,
        auto_delete: bool = True,
    ):
        """
        Initializes an instance of the class.

        Args:
            instruction (str): The instruction for the assistant.
            model (str): The model to be used by the assistant.
            use_retrieval (bool, optional): Whether to use retrieval-based tools. Defaults to False.
            use_code_interpreter (bool, optional): Whether to use code interpretation tools. Defaults to False.
            file_ids (List[str], optional): The list of file IDs. Defaults to None.
            functions (List[Function], optional): The list of functions. Defaults to None.
            assistant_name (str, optional): The name of the assistant. Defaults to "AI Assistant".
            assistant_description (str, optional): The description of the assistant. Defaults to "An AI Assistant".
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            auto_delete (bool, optional): Whether to enable auto deletion. Defaults to True.
        """
        self.client = Client()
        self.instruction = instruction
        self.model = model
        self.use_retrieval = use_retrieval
        self.use_code_interpreter = use_code_interpreter
        self.file_ids = file_ids
        self.functions = functions
        self.assistant_name = assistant_name
        self.assistant_description = assistant_description
        self.tools = [
            {"type": "function", "function": f.to_dict()} for f in self.functions
        ] if self.functions else []
        if self.use_retrieval:
            self.tools.append({"type": "retrieval"})
        if self.use_code_interpreter:
            self.tools.append({"type": "code_interpreter"})
        self.assistant = self.client.beta.assistants.create(
            name=self.assistant_name,
            description=self.assistant_description,
            instructions=self.instruction,
            model=self.model,
            tools=self.tools,
            file_ids=self.file_ids if self.file_ids else [],
        )
        self.threads = []
        self.conversation = Conversation(messages=[])
        self.verbose = verbose
        self.auto_delete = auto_delete

    def delete_assistant_file_by_id(self, file_id: str):
        """
        Deletes an assistant file by its ID.

        :param file_id: The ID of the file to be deleted.
        :type file_id: str
        :return: The status of the file deletion.
        :rtype: dict
        """
        file_deletion_status = self.client.beta.assistants.files.delete(
            assistant_id=self.assistant.id, file_id=file_id
        )
        return file_deletion_status

    def create_thread(self) -> Thread:
        """
        Creates a new thread and adds it to the list of threads.

        Returns:
            Thread: The newly created thread.
        """
        thread = self.client.beta.threads.create()
        self.threads.append(thread)
        return thread

    def create_tool_outputs(self, run: Run) -> List[dict]:
        """
        Generates a list of tool outputs for a given run.

        Args:
            run (Run): The run object containing the required action and tool outputs.

        Returns:
            List[dict]: A list of tool outputs, each containing the tool call ID and the output.

        Raises:
            None
        """
        tool_outputs = []
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            tool_found = False
            function_name = tool.function.name
            if tool.function.arguments:
                function_arguments = json.loads(tool.function.arguments)
            else:
                function_arguments = {}
            call_id = tool.id
            function_call = FunctionCall(
                call_id=call_id, name=function_name, arguments=function_arguments
            )
            for function in self.functions:
                if function.name == function_name:
                    tool_found = True
                    if self.verbose:
                        random_color = random.choice(PRINT_COLORS)
                        print(f'\n{random_color}{function_name} function has called by assistant with the following arguments: {function_arguments}')
                    response = function.run_catch_exceptions(
                        function_call=function_call
                    )
                    if self.verbose:
                        random_color = random.choice(PRINT_COLORS)
                        print(f"{random_color}Function {function_name} responsed: {response}")
                    tool_outputs.append(
                        {
                            "tool_call_id": call_id,
                            "output": response,
                        }
                    )
            if not tool_found:
                if self.verbose:
                    random_color = random.choice(PRINT_COLORS)
                    print(f"{random_color}Function {function_name} alled by assistant not found")
                tool_outputs.append(
                    {
                        "tool_call_id": call_id,
                        "output": f"Function {function_name} not found",
                    }
                )
        return tool_outputs

    def get_required_functions_names(self, run: Run) -> List[str]:
        """
        Return a list of required function names for a given run.

        Args:
            run (Run): The run object containing the required action and tool outputs.

        Returns:
            List[str]: A list of function names extracted from the tool calls.
        """
        function_names = []
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            function_names.append(tool.function)
        return function_names

    def create_conversation(self, thread_id: str):
        """
        Creates a conversation based on the given thread ID.

        Args:
            thread_id (str): The ID of the thread.

        Returns:
            str: The printed conversation.
        """
        messages = self.client.beta.threads.messages.list(thread_id=thread_id).data
        for message in messages:
            self.conversation.messages.append(
                Message(
                    thread_id=thread_id,
                    role=message.role,
                    content=self.format_message(message=message),
                    file_ids=message.file_ids,
                )
            )
        return self.conversation.print_conversation()
    
    def list_files(self):
        """
        Retrieves a list of files from the client.

        :return: A list of file data.
        """
        return self.client.files.list().data
    
    def create_file(self, filename: str, file_id: str):
        """
        Create a file from the given file ID and save it with the given filename.
        
        Parameters:
            filename (str): The name of the file to be created.
            file_id (str): The ID of the file to retrieve the content from.
        
        Returns:
            None
        
        Raises:
            FileNotFoundError: If the file ID does not exist.
        """
        content = self.client.files.content(file_id)
        with open(filename.split("/")[-1], 'w', encoding='utf-8') as file:
            file.write(content)

    def upload_file(self, filename: str) -> str:
        """
        Uploads a file to the server.

        Args:
            filename (str): The name of the file to be uploaded.

        Returns:
            str: The ID of the uploaded file.
        """
        file = self.client.files.create(
            file=open(filename, "rb"),
            purpose='assistants'
        )
        return file.id
    
    def delete_file(self, file_id: str) -> bool:
        """
        Deletes a file with the given file ID.

        Parameters:
            file_id (str): The ID of the file to be deleted.

        Returns:
            bool: True if the file was successfully deleted, False otherwise.
        """
        file_deletion_status = self.client.beta.assistants.files.delete(
            assistant_id=self.assistant.id,
            file_id=file_id
            )
        return file_deletion_status.deleted

    def format_message(self, message: ThreadMessage) -> str:
        """
        Formats the message content by replacing annotations with their corresponding indices and generating citations for files.
        
        Parameters:
            message (ThreadMessage): The message object containing the content to be formatted.
        
        Returns:
            str: The formatted message content.
        """
        if getattr(message.content[0], "text", None) is not None:
            message_content = message.content[0].text
        else:
            message_content = message.content[0]
        annotations = message_content.annotations
        citations = []
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(
                annotation.text, f" [{index}]"
            )
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = self.client.files.retrieve(file_citation.file_id)
                citations.append(
                    f"[{index}] {file_citation.quote} from {cited_file.filename}"
                )
            elif file_path := getattr(annotation, "file_path", None):
                cited_file = self.client.files.retrieve(file_path.file_id)
                citations.append(
                    f"[{index}] file: {cited_file.filename} is downloaded"
                )
                self.create_file(filename=cited_file.filename, file_id=cited_file.id)

        message_content.value += "\n" + "\n".join(citations)
        return message_content.value

    def extract_run_message(self, run: Run, thread_id: str) -> str:
        """
        Extracts the run message from the specified thread.

        Args:
            run (Run): The run object.
            thread_id (str): The ID of the thread.

        Returns:
            str: The extracted run message, or "Assistant: No message found" if no message is found.
        """
        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id,
        ).data
        for message in messages:
            if message.run_id == run.id:
                return f"{message.role}: " + self.format_message(message=message)
        return "Assistant: No message found"

    def create_response(
        self,
        thread_id: str,
        content: str,
        message_files: List[str] = None,
        run_instructions: str = None,
    ) -> str:
        """
        Creates a response message in a specified thread.

        Args:
            thread_id (str): The ID of the thread where the message will be created.
            content (str): The content of the message.
            message_files (List[str], optional): List of file IDs to attach to the message. Defaults to None.
            run_instructions (str, optional): The instructions for running the thread. Defaults to None.

        Returns:
            str: The response message extracted from the run.
        """
        self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content,
            file_ids=message_files if message_files else [],
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistant.id,
            instructions=run_instructions,
        )
        with yaspin(text="Loading", color="yellow"):
            while run.status != "completed":
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id, run_id=run.id
                )
                if run.status == "failed":
                    raise RuntimeError(f"Run failed with the following error {run.last_error}")
                if run.status == "expired":
                    raise RuntimeError(
                        f"Run expired when calling {self.get_required_functions_names(run=run)}"
                    )
                if run.status == "requires_action":
                    tool_outputs = self.create_tool_outputs(run=run)
                    run = self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=tool_outputs,
                    )
                if self.verbose:
                    random_color = random.choice(PRINT_COLORS)
                    print(f"\n{random_color}Run status: {run.status}")
                time.sleep(0.5)
        return "\n" + self.extract_run_message(run=run, thread_id=thread_id)
    
    def chat(self, file_ids: List[str] = None):
        """
        Chat function that allows the user to interact with the assistant.

        Parameters:
        - file_ids (List[str]): Optional list of file IDs to attach to the messages.
        
        Returns:
        - None
        """
        thread = self.create_thread()
        user_input = ""
        while user_input != "bye" and user_input != "exit":
            user_input = input("\033[32mYou (type bye to quit): ")
            message = self.create_response(
            thread_id=thread.id, content=user_input, message_files=file_ids
            )
            print(f"\033[33m{message}")
        if self.auto_delete:
            if file_ids:
                for file in file_ids:
                    self.delete_file(file_id=file)
            self.client.beta.threads.delete(thread_id=thread.id)
            self.client.beta.assistants.delete(assistant_id=self.assistant.id)
