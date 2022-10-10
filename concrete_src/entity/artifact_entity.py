from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact", 
[ "raw_data_file_path", "is_ingested", "message" ])