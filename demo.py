from concrete_src.logger import logging
from concrete_src.exception import ConcreteException
from concrete_src.config.configuartion import Configuration
from concrete_src.pipeline.pipeline import Pipeline
import sys



def main():
    try:
        config = Configuration()
        pipeline = Pipeline(config)
        pipeline.run_pipeline()
    except Exception as e:
        raise ConcreteException(e, sys) from e
        print(e)
        logging.error(f"{e}")

if __name__=="__main__":
    main()