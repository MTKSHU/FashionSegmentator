#!/bin/bash

cd fashion_segmentator/


# Apply database migrations
echo "Apply database migrations"
python manage.py makemigrations
python manage.py migrate

# Collect static files
echo "Collect static files"
python manage.py collectstatic --noinput

echo "from django.contrib.auth.models import User; User.objects.create_superuser('admin', 'admin@example.com', 'adminpass')" | python manage.py shell

#Start Unicorn Process
echo Starting Unicorn.

pwd
exec gunicorn -c gunicorn.config.py fashion_segmentator.wsgi:application