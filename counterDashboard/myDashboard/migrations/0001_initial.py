# Generated by Django 4.2.7 on 2023-11-16 13:49

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PressedButton',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('press_date', models.DateTimeField(auto_now_add=True)),
                ('result_number', models.IntegerField()),
            ],
        ),
    ]