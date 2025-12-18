"""
Download emails from Outlook.com via Microsoft Graph API
Uses .env file for configuration
Saves emails as .eml files with RESTORED original headers
UPDATED: Replaces headers (From/To/Date) and removes forwarding note

Requirements:
    pip install msal requests python-dotenv
"""

import os
import requests
import msal
import json
from pathlib import Path
from datetime import datetime
import time
from dotenv import load_dotenv
import logging
import re
import email
from email import policy
from email.parser import BytesParser
from email.message import EmailMessage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OutlookGraphDownloader:
    """Download emails from Outlook.com using Microsoft Graph API"""
    
    def __init__(self):
        """Initialize with configuration from .env file"""
        
        # Load environment variables
        load_dotenv()
        
        # Required configuration
        self.client_id = os.getenv('AZURE_CLIENT_ID')
        self.tenant_id = os.getenv('AZURE_TENANT_ID', 'consumers')
        self.client_secret = os.getenv('AZURE_CLIENT_SECRET')
        
        # Optional configuration
        self.output_dir = os.getenv('OUTPUT_DIR', './downloaded_emails')
        self.max_emails = int(os.getenv('MAX_EMAILS', '100'))
        self.filter_query = os.getenv('FILTER_QUERY', None)
        self.user_email = os.getenv('USER_EMAIL', None)
        
        # Validate required config
        if not self.client_id:
            raise ValueError("AZURE_CLIENT_ID not found in .env file")
        
        # Set up authority URL
        if self.tenant_id == 'consumers' or self.tenant_id == 'common':
            self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        else:
            self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        
        # Scopes for Graph API
        if self.client_secret:
            self.scopes = ["https://graph.microsoft.com/.default"]
        else:
            self.scopes = ["Mail.Read", "offline_access"]
        
        # Initialize MSAL app
        if self.client_secret:
            self.app = msal.ConfidentialClientApplication(
                client_id=self.client_id,
                client_credential=self.client_secret,
                authority=self.authority
            )
            logger.info("Initialized as confidential client (daemon)")
        else:
            self.app = msal.PublicClientApplication(
                client_id=self.client_id,
                authority=self.authority,
                client_capabilities=None
            )
            logger.info("Initialized as public client (interactive)")
        
        self.access_token = None
    
    def authenticate(self):
        """Authenticate and get access token"""
        
        logger.info("Starting authentication...")
        
        if self.client_secret:
            logger.info("Using client credentials flow (confidential client)")
            try:
                result = self.app.acquire_token_for_client(scopes=self.scopes)
                
                if "access_token" in result:
                    self.access_token = result['access_token']
                    logger.info("✓ Authentication successful (client credentials)")
                    return True
                else:
                    error_desc = result.get('error_description', result.get('error', 'Unknown error'))
                    logger.error(f"Authentication failed: {error_desc}")
                    return False
                    
            except Exception as e:
                logger.error(f"Exception during authentication: {e}")
                return False
        
        # Public client flow
        accounts = self.app.get_accounts()
        if accounts:
            logger.info("Found account in cache, attempting silent authentication")
            result = self.app.acquire_token_silent(self.scopes, account=accounts[0])
            if result and "access_token" in result:
                self.access_token = result['access_token']
                logger.info("✓ Authentication successful (from cache)")
                return True
        
        flow = self.app.initiate_device_flow(scopes=self.scopes)
        
        if "user_code" not in flow:
            logger.error(f"Failed to create device flow: {json.dumps(flow, indent=2)}")
            return False
        
        print("\n" + "="*60)
        print(flow['message'])
        print("="*60 + "\n")
        
        logger.info("Waiting for user to authenticate...")
        
        result = self.app.acquire_token_by_device_flow(flow)
        
        if "access_token" in result:
            self.access_token = result['access_token']
            logger.info("✓ Authentication successful")
            return True
        else:
            logger.error(f"Authentication failed: {result.get('error_description', 'Unknown error')}")
            return False
    
    def get_messages(self):
        """Get messages from inbox using Graph API and limit to the first 5."""
        
        if not self.access_token:
            logger.error("Not authenticated. Please call authenticate() first.")
            return []
        
        logger.info("Fetching messages from inbox...")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        # Build Graph API URL
        if self.client_secret and self.user_email:
            url = f"https://graph.microsoft.com/v1.0/users/{self.user_email}/mailFolders/inbox/messages"
            logger.info(f"Using application permissions for user: {self.user_email}")
        else:
            url = "https://graph.microsoft.com/v1.0/me/mailFolders/inbox/messages"
            logger.info("Using delegated permissions (/me endpoint)")
        
        params = {
            '$select': 'id,subject,from,toRecipients,ccRecipients,receivedDateTime,hasAttachments,importance',
            '$top': 6,  # Explicitly set to 5 messages only
            '$orderby': 'receivedDateTime DESC'
        }
        
        if self.filter_query:
            params['$filter'] = self.filter_query
            logger.info(f"Applying filter: {self.filter_query}")
        
        messages = []
        page_count = 0
        
        # We only want one page, so the while loop should run once.
        # We use a placeholder for 'url' that will be set to None after the first request.
        
        while url: # This ensures the first request runs
            try:
                response = requests.get(
                    url, 
                    headers=headers, 
                    params=params if page_count == 0 else None
                )
                
                if response.status_code != 200:
                    logger.error(f"Error fetching messages: {response.status_code}")
                    logger.error(response.text)
                    break
                
                data = response.json()
                batch_messages = data.get('value', [])
                messages.extend(batch_messages)
                
                page_count += 1
                logger.info(f"Fetched page {page_count}: {len(messages)} messages total")
                
                # --- CHANGE 1: Stop the loop immediately ---
                # Stop the iteration by clearing 'url' and optionally breaking out.
                url = None
                break
                # -------------------------------------------
                
            except Exception as e:
                logger.error(f"Error fetching messages: {e}")
                break
        
        # Ensure we only return a maximum of 5 messages in case $top was ignored or logic was flawed
        messages = messages[:5]
        
        logger.info(f"✓ Retrieved {len(messages)} messages")
        return messages
    
    def get_message_mime(self, message_id):
        """Get message in MIME format"""
        
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        
        if self.client_secret and self.user_email:
            url = f"https://graph.microsoft.com/v1.0/users/{self.user_email}/messages/{message_id}/$value"
        else:
            url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/$value"
        
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.content
            else:
                logger.warning(f"Error fetching MIME for message {message_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Exception fetching MIME: {e}")
            return None
    
    def extract_original_metadata(self, mime_content):
        """
        Extract original From, To, and Date from forwarded email note
        
        Returns:
            Dictionary with original_from, original_to, original_date, or None if not found
        """
        
        try:
            msg = BytesParser(policy=policy.default).parsebytes(mime_content)
            
            # Get the body text
            body_text = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            body_text = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            break
                        except:
                            pass
            else:
                try:
                    body_text = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                except:
                    body_text = str(msg.get_payload())
            
            if not body_text:
                return None
            
            metadata = {}
            
            # Extract From (full format: Name <email@domain.com>)
            from_match = re.search(r'^From:\s*(.+)$', body_text, re.MULTILINE)
            if from_match:
                metadata['original_from'] = from_match.group(1).strip()
            
            # Extract To
            to_match = re.search(r'^To:\s*(.+)$', body_text, re.MULTILINE)
            if to_match:
                metadata['original_to'] = to_match.group(1).strip()
            
            # Extract Date
            date_match = re.search(r'^Date:\s*(.+)$', body_text, re.MULTILINE)
            if date_match:
                metadata['original_date'] = date_match.group(1).strip()
            
            if 'original_from' in metadata:
                logger.debug(f"Extracted original metadata: {metadata}")
                return metadata
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not extract original metadata: {e}")
            return None
    
    def remove_forwarding_note(self, body_text):
        """
        Remove the forwarding note from email body
        
        Removes everything from [FORWARDED EMAIL...] to ---Original Message---
        """
        
        # Pattern to match the forwarding note
        pattern = r'\[FORWARDED EMAIL.*?---Original Message---\s*\n*'
        
        cleaned_body = re.sub(pattern, '', body_text, flags=re.DOTALL)
        
        return cleaned_body.strip()
    
    def restore_original_email(self, mime_content, original_metadata):
        """
        Restore original email headers and remove forwarding note
        
        Args:
            mime_content: Original MIME content as bytes
            original_metadata: Dict with original_from, original_to, original_date
        
        Returns:
            Modified MIME content as bytes with restored headers
        """
        
        try:
            # Parse the original message
            msg = BytesParser(policy=policy.default).parsebytes(mime_content)
            
            # Replace headers with original values
            if 'original_from' in original_metadata:
                msg.replace_header('From', original_metadata['original_from'])
            
            if 'original_to' in original_metadata:
                msg.replace_header('To', original_metadata['original_to'])
            
            if 'original_date' in original_metadata:
                msg.replace_header('Date', original_metadata['original_date'])
            
            # Remove the forwarding note from body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            cleaned_body = self.remove_forwarding_note(body)
                            part.set_payload(cleaned_body.encode('utf-8'))
                            # Update encoding if needed
                            del part['Content-Transfer-Encoding']
                            part.add_header('Content-Transfer-Encoding', '8bit')
                        except:
                            pass
            else:
                # Simple message
                try:
                    body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                    cleaned_body = self.remove_forwarding_note(body)
                    msg.set_payload(cleaned_body.encode('utf-8'))
                    del msg['Content-Transfer-Encoding']
                    msg.add_header('Content-Transfer-Encoding', '8bit')
                except:
                    pass
            
            # Convert back to bytes
            return msg.as_bytes()
            
        except Exception as e:
            logger.error(f"Error restoring original email: {e}")
            return mime_content
    
    def extract_email_address(self, email_header):
        """Extract just the email address from a header like 'Name <email@domain.com>'"""
        
        if not email_header:
            return 'unknown'
        
        match = re.search(r'<([^>]+)>', email_header)
        if match:
            return match.group(1).strip()
        
        return email_header.strip()
    
    def parse_email_date(self, date_str):
        """Parse email date string to datetime object"""
        
        if not date_str:
            return None
        
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str)
        except:
            try:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except:
                return None
    
    def sanitize_filename(self, filename, max_length=250):
        """Sanitize filename to be filesystem-safe"""
        
        invalid_chars = '<>:"/\\|?* '
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        filename = filename.strip('. ')
        
        if len(filename) > max_length:
            filename = filename[:max_length]
        
        return filename if filename else 'unnamed'
    
    def download_messages(self):
        """Download messages and save into ONE emails.json file"""

        if not self.access_token:
            if not self.authenticate():
                return {'success': 0, 'failed': 0, 'total': 0}

        messages = self.get_messages()

        if not messages:
            logger.info("No messages to download")
            return {'success': 0, 'failed': 0, 'total': 0}

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {len(messages)} message(s) into a single JSON file...")
        logger.info("=" * 60)

        json_output = []
        successful = 0
        failed = 0

        for i, message in enumerate(messages, 1):
            try:
                msg_id = message['id']
                subject = message.get('subject', '')

                logger.info(f"[{i}/{len(messages)}] Processing: {subject[:80]}")

                # Fetch MIME
                mime_content = self.get_message_mime(msg_id)
                if not mime_content:
                    logger.warning("  ✗ Failed to retrieve MIME content")
                    failed += 1
                    continue

                # Extract original metadata
                original_metadata = self.extract_original_metadata(mime_content)
                if original_metadata:
                    logger.info("  🔄 Restoring original metadata...")
                    mime_content = self.restore_original_email(mime_content, original_metadata)

                # Base64 encode MIME for JSON storage
                import base64
                mime_b64 = base64.b64encode(mime_content).decode('utf-8')

                entry = {
                    "id": msg_id,
                    "subject": subject,
                    "from": message.get('from', {}).get('emailAddress', {}).get('address', ''),
                    "to": [r['emailAddress']['address'] for r in message.get('toRecipients', [])],
                    "receivedDateTime": message.get('receivedDateTime', ''),
                    "restoredMetadata": original_metadata if original_metadata else None,
                    "mime_base64": mime_b64
                }

                json_output.append(entry)
                successful += 1

            except Exception as e:
                logger.error(f"  ✗ Error processing message: {e}")
                failed += 1

        # Save single JSON file
        json_path = output_path / "emails.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2)

        logger.info("")
        logger.info("=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✓ Successful: {successful}")
        logger.info(f"✗ Failed: {failed}")
        logger.info(f"Total: {len(messages)}")
        logger.info(f"Saved to: {json_path}")

        return {
            'success': successful,
            'failed': failed,
            'total': len(messages)
        }



def main():
    """Main entry point"""
    
    print("="*60)
    print("OUTLOOK.COM EMAIL DOWNLOADER (Graph API)")
    print("Restores original headers and removes forwarding notes")
    print("="*60)
    
    try:
        downloader = OutlookGraphDownloader()
        stats = downloader.download_messages()
        
        if stats['failed'] > 0:
            exit(1)
        else:
            exit(0)
            
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("\nPlease ensure your .env file contains:")
        logger.error("  AZURE_CLIENT_ID=<your-client-id>")
        logger.error("  AZURE_TENANT_ID=consumers")
        logger.error("  OUTPUT_DIR=./downloaded_emails")
        exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
