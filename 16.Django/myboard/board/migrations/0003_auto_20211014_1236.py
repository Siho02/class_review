# Generated by Django 2.2.5 on 2021-10-14 03:36

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('board', '0002_auto_20211011_1222'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='post',
            options={'ordering': ['-id']},
        ),
    ]
