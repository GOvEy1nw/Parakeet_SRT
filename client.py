import requests
import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Client to request transcription from the Parakeet ASR service."
    )
    parser.add_argument("input_path", help="Path to the input media file.")
    parser.add_argument(
        "--output", required=True, help="Path to save the output SRT file."
    )

    args = parser.parse_args()

    # Get absolute paths to ensure the server can find the files, especially in Docker.
    input_path_abs = os.path.abspath(args.input_path)
    output_path_abs = os.path.abspath(args.output)

    server_url = "http://localhost:5000/transcribe"
    payload = {"input_path": input_path_abs, "output_path": output_path_abs}

    print(f"Sending request to {server_url} with payload: {payload}")

    try:
        response = requests.post(
            server_url, json=payload, timeout=None
        )  # Timeout=None for long transcriptions
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        response_data = response.json()
        print(f"Server response: {response_data}")

        if response_data.get("status") != "success":
            sys.exit(1)  # Exit with an error code if server reported an error

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the transcription service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
