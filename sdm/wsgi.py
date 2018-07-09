"""
WSGI config for sdm project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.0/howto/deployment/wsgi/
"""

import os, sys

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sdm.settings")

sys.path.append('/var/www/html/gemini/sdm/')

sys.path.append('/var/www/html/gemini/django/lib/python3.6/site-packages')

application = get_wsgi_application()
