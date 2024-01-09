""" 
This module contains the Function class, which represents a function in the assistant.
"""

from abc import ABC, abstractmethod
import traceback
from typing import Optional, Dict, List
from pydantic import BaseModel


class Property(BaseModel):
    """ Represents a property of a function.

    Args:
        name: The name of the property.
        type: The type of the property.
        required: Whether the property is required or not.
        description: The description of the property.
        
    Returns:
        None
    """
    name: str
    type: str
    required: bool = True
    description: Optional[str] = None

class InvalidParametersException(Exception):
    """ 
    Represents an invalid parameters exception.

    Args:
        message: The message of the exception.

    Returns:
        None
    """
    message: str


class MissingParametersException(Exception):
    """
    Represents a missing parameters exception.

    Args:
        message: The message of the exception.

    Returns:
        None
    """
    message: str

class UnexpectedParametersException(Exception):
    """
    Represents an unexpected parameters exception.

    Args:
        message: The message of the exception.

    Returns:
        None
    """
    message: str

class FunctionCall(BaseModel):
    """
    Represents a function call.

    Args:
        call_id: The ID of the function call.
        name: The name of the function.
        arguments: The arguments of the function call.

    Returns:
        None
    """
    call_id: str
    name: str
    arguments: Optional[Dict] = None


class Function(BaseModel, ABC):
    """
    Represents a function.

    Args:
        name: The name of the function.
        description: The description of the function.
        parameters: The parameters of the function.

    Returns:
        None
    """
    name: str
    description: Optional[str] = None
    parameters: Optional[List[Property]] = None

    def to_dict(self):
        """
        Returns a dictionary representation of the object.

        If parameters is None, returns a dictionary with the following keys:
        - name: The name of the object.
        - description: The description of the object.
        - parameters: A dictionary with the following keys:
            - type: The type of the parameters (always set to "object").
            - properties: An empty dictionary.
            - required: An empty list.

        If parameters is not None, returns a dictionary with the following keys:
        - name: The name of the object.
        - description: The description of the object.
        - parameters: A dictionary with the following keys:
            - type: The type of the parameters (always set to "object").
            - properties: A dictionary where each key is a parameter name and each value is a dictionary with the following keys:
                - type: The type of the parameter.
                - description: The description of the parameter.
            - required: A list of parameter names that are required.

        Returns:
            dict: A dictionary representation of the object.
        """
        if self.parameters is None:
            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.name: {"type": p.type, "description": p.description}
                    for p in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required],
            },
        }

    def run(self, function_call: FunctionCall = None):
        """
        Runs the function with the given function call.

        Args:
            function_call (FunctionCall, optional): The function call object containing the arguments for the function. Defaults to None.

        Raises:
            Exception: If the function call arguments are empty and the parameters are not None.
            Exception: If the function call arguments are empty and the parameters are None.
            Exception: If the function call arguments are not empty and the parameters are None.
            Exception: If a required parameter is missing in the function call arguments.

        Returns:
            The result of the function call.
        """
        if not function_call.arguments and self.parameters:
            raise MissingParametersException("Missing parameters")

        if function_call.arguments and not self.parameters:
            raise UnexpectedParametersException("Unexpected parameters")

        if self.parameters:
            missing_parameters = [p.name for p in self.parameters if p.required and p.name not in function_call.arguments]
            if missing_parameters:
                raise MissingParametersException(f"Missing parameter(s): {', '.join(missing_parameters)}")

        return self.func(**function_call.arguments)

    def run_catch_exceptions(self, function_call: FunctionCall = None):
        """
        Runs the given function call and catches any exceptions that are raised.
        
        Parameters:
            function_call (FunctionCall, optional): The function call to be executed. Defaults to None.
        
        Returns:
            str: The string representation of the exception if one is raised, otherwise the result of the function call.
        """
        try:
            return self.run(function_call=function_call)
        except Exception as e:
            return f'{e}\n{traceback.format_exc()}'

    @abstractmethod
    def func(self, **kwargs):
        """
        A description of the entire function, its parameters, and its return types.

        Args:
            **kwargs: The arguments for the function call.

        """
