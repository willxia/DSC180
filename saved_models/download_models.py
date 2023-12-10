import gdown
import zipfile
import os

def download_google_drive(file_id, output_file):
    # Extract file ID from the Google Drive link

    # Get the direct download link using the file ID
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # Download the zip file
    gdown.download(download_url, output_file, quiet=False)

    print("Download and extraction complete.")

if __name__ == "__main__":
    bert_model_gdrive_id = "1x_mEn9idwcN-vUSGziKpn4v9JexvTZJl"
    logreg_model_gdrive_id = "1XX_SuO2hym7MhNopcRWE4njF5T7uP60C"
    download_google_drive(bert_model_gdrive_id, "bert_model.zip")
    download_google_drive(logreg_model_gdrive_id, "logreg_model.sav")
    
    with zipfile.ZipFile("bert_model.zip", 'r') as zip_ref:
        zip_ref.extractall()
    # Remove the downloaded zip file
    os.remove("bert_model.zip")

