lint:
	yapf -i *.py

web:
	python backend/app.py

guni:
	gunicorn -b 0.0.0.0:5000 backend.wsgi:app --reload
