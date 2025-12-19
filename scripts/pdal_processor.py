import argparse
import json
import pdal
import sys
import os

def process_pipeline(input_file, output_file, pipeline_json_str):
    """
    Executes a PDAL pipeline after substituting placeholder filenames.
    """
    print(f"Starting PDAL processing.")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # Replace placeholders
    pipeline_str = pipeline_json_str.replace("__INPUT_FILE__", input_file)
    pipeline_str = pipeline_str.replace("__OUTPUT_FILE__", output_file)
    
    # Basic validation
    try:
        pipeline_json = json.loads(pipeline_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Pipeline content: {pipeline_str}")
        sys.exit(1)

    # Execute Pipeline
    try:
        pipeline = pdal.Pipeline(pipeline_str)
        count = pipeline.execute()
        print(f"Pipeline executed successfully. Processed {count} points.")
        print(f"Metadata: {pipeline.metadata}")
    except RuntimeError as e:
        print(f"PDAL Pipeline Runtime Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a PDAL pipeline.")
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--pipeline", required=True, help="JSON pipeline definition")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    process_pipeline(args.input, args.output, args.pipeline)
