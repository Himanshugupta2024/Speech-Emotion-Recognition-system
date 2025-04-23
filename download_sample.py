import os
import requests
import zipfile
import io

def download_sample_audio():
    """
    Download a sample audio file for testing.
    """
    # Create directory for sample data
    os.makedirs('sample_data', exist_ok=True)
    
    # URL for a sample audio file from the RAVDESS dataset
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    
    print(f"Downloading sample audio files from {url}...")
    
    try:
        # Download the zip file
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract the zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Extract only a few files to save space
            for file in zip_ref.namelist():
                if file.endswith('.wav') and 'Actor_01' in file:
                    zip_ref.extract(file, 'sample_data')
        
        print("Sample audio files downloaded successfully!")
        print("Files are located in the 'sample_data' directory.")
        
        # List the downloaded files
        files = [f for f in os.listdir('sample_data/Actor_01') if f.endswith('.wav')]
        print(f"Downloaded {len(files)} files:")
        for file in files[:5]:
            print(f"  - {file}")
        if len(files) > 5:
            print(f"  - ... and {len(files) - 5} more")
    
    except Exception as e:
        print(f"Error downloading sample audio files: {e}")

if __name__ == "__main__":
    download_sample_audio()
