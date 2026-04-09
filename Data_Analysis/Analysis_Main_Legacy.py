"""
Use this entry point only when you intentionally want the older workflow:
edit the in-script defaults in Analysis_Main_Engine.py and run without --config.
"""

from Analysis_Main_Engine import main, reset_user_settings_to_defaults


if __name__ == "__main__":
    reset_user_settings_to_defaults()
    print("Running in legacy mode using the in-script default settings from Analysis_Main_Engine.py.")
    main()
