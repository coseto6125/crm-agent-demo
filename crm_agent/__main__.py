import os

from crm_agent.app import create_app

if __name__ == "__main__":
    create_app(os.environ.get("CRM_DB", "crm.db")).run(host="127.0.0.1", port=8000, single_process=True)
