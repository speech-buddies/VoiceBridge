"""User Profile Manager module."""

from datetime import datetime
from typing import Any, Dict, List, Tuple
import uuid

import CredentialManager
import EncryptionManager
import StorageManager

# Exceptions.
class ProfileCreationException(Exception):
    """Exception raised for profile creation failure."""

    def __init__(self, message: str="Failed to create profile."):
        super().__init__(message)

class DataNotFoundException(Exception):
    """Exception raised for when data can not be found."""

    def __init__(self, message: str="Data not found exception."):
        super().__init__(message)

# Class defintion.
class UserProfileManager:
    """
    User Profile Manager module.

    Raises:
        ProfileCreationException: Raised for failures in creating a profile.
        DataNotFoundException: Requested data that this module handles is not found.

    """

    # Profiles maps profile ID to profile meta data.
    profiles: Dict[str, Dict]

    # Preferences maps profile ID to user's preference data.
    preferences: Dict[str, PreferenceData]

    # Consent log. Each entry holds time of log, profile Id and the cosent value.
    # Consent value is True for consent given and false otherwise.
    consent_log: List[Tuple[datetime, str, bool]]


    def __init__(self):
        """Constructor."""
        self.profiles = {}
        self.preferences = {}
        self.consent_log = []

    def create_profile(self, user_token: str, init_data: Dict[str, Any]) -> str:
        """
        Creates a profile and adds to internal database.

        Args:
            user_token (str): User session token
            init_data (Dict[str, Any]): User profile meta data.

        Raises:
            ProfileCreationException: Raised for failures in creating a profile.

        Returns:
            str: Profile ID
        """

        # Validate user has an active session.
        if not credentialManager.validate_token(user_token):
            msg = "Failed to create profile. User token is invalid."
            raise ProfileCreationException(msg)

        # Validate user does not have a profile already.
        profile_id = str(uuid.uuid4())
        if profile_id in self.profiles.keys():
            msg = f"User already has profile. Profile ID: {profile_id}"
            raise ProfileCreationException(msg)

        # Encode profile initialization data. Format: '|field=data|'
        profile_data = ""
        for field, data in init_data.items():
            profile_data += f"|{field}={data}|"

        # Encrypt user data.
        cipher_data = encryptionManager.encrypt_data(profile_data, KEYID)

        # Save profile.
        self.profiles[profile_id] = init_data
        self.preferences[profile_id] = init_data.preferences
        storageManager.store_profile(cipher_data, profile_id)

        return profile_id

    def load_preferences(self, profileID: str) -> PreferenceData:
        """
        Loads profile preferences.

        Args:
            profileID (str): Profile ID associated tot he preference data to load.

        Raises:
            DataNotFoundException: Preference data is not found for the profile ID.

        Returns:
            PreferenceData: Preference data associated to the profile ID.
        """

        if not profileID in self.preferences.keys():
            raise DataNotFoundException

        return self.preferences[profileID]

    def save_consent(self, profile_id: str, consent_flag: bool) -> bool:
        """
        Update profile consent to store user data.

        Args:
            profile_id (str): Profile ID to update consent value.
            consent_flag (bool): True for consent to store data. False otherwise. 

        Returns:
            bool: True for successful udpate. False otherwise.
        """

        status = True
        self.consent_log.append((datetime.now(), profile_id, consent_flag))

        # Clear user preference data if user removes consent.
        if consent_flag is False and profile_id in self.preferences.keys():
            self.preferences[profile_id] = {}

        return status
