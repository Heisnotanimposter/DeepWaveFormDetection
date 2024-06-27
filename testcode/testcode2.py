from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials

# Define the scope
scope = ['https://www.googleapis.com/auth/drive.readonly']

# Authenticate using the service account key
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'service_account_key.json', scope)

# Build the service
service = build('drive', 'v3', credentials=credentials)

# Call the Drive v3 API
results = service.files().list(
    fields="*",
    corpora='drive',
    supportsAllDrives=True,
    driveId="YOUR_DRIVE_ID",
    includeItemsFromAllDrives=True
).execute()

items = results.get('files', [])

if not items:
    print('No files found.')
else:
    print('Files:')
    for item in items:
        print(u'{0} ({1})'.format(item['name'], item['id']))
