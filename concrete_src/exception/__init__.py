import os
import sys
from concrete_src.logger import logging

class ConcreteException(Exception):
    
    def __init__(self, error_message:Exception,error_detail:sys):
        """
        Input: 
        error_message : Exception object
        error_detail : object of sys module
        """
        super().__init__(error_message)
        self.error_message=ConcreteException.get_detailed_error_message(error_message=error_message,
                                                                       error_detail=error_detail
                                                                        )
        logging.error(self.error_message)

    @staticmethod
    def get_detailed_error_message(error_message:Exception,error_detail:sys)->str:
        """
        error_message: Exception object
        error_detail: object of sys module
        """
        _,_ ,exec_tb = error_detail.exc_info()
        exception_block_line_number = exec_tb.tb_frame.f_lineno
        try_block_line_number = exec_tb.tb_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename
        error_message = f"""
        Error occured in : 
        [ {file_name} ] at 
        try block line number: [{try_block_line_number}] and exception block line number: [{exception_block_line_number}] 
        error message: [{error_message}]
        """
        return error_message

    def __str__(self):
        return self.error_message


    def __repr__(self) -> str:
        return ConcreteException.__name__.str()
