from concrete_src.logger import logging
from concrete_src.exception import ConcreteException
import sys



def main():
    try:
        logging.info("Checking log.")
        raise Exception("Checking Exception")
    except Exception as e:
        raise ConcreteException(e, sys) from e
        print(e)
        logging.error(f"{e}")

if __name__=="__main__":
    main()