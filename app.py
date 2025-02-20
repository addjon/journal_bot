from flask import Flask, request
import requests
import os
from dotenv import load_dotenv
import re
from datetime import datetime
import logging
import sqlite3
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload
#from langchain import LLMChain
#from langchain.prompts import PromptTemplate
#from langchain.llms import OpenAI  # Или другой провайдер
from openai import OpenAI

# Загрузка переменных окружения из файла .env
load_dotenv()

# Получение токена
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
URL = f"https://api.telegram.org/bot{TOKEN}/"

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(
    filename='app.log',  # Имя файла для логов
    level=logging.DEBUG,  # Уровень логирования
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

with open('config.json', 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

# Обработка входящих сообщений от Telegram
@app.route(f"/webhook/{TOKEN}", methods=["POST"])
def webhook():
    data = request.json
    logging.info(f"Получены данные: {data}")
    try:
        chat_id = data["message"]["chat"]["id"]
        message_text = data["message"]["text"]

        # Обработка сообщения
        parsed_data, success = parse_telegram_message(message_text)
        if success:
            # Сохраняем данные в базу данных
            save_data_to_database(parsed_data)
            logging.info(f"Распарсенные данные: {parsed_data}")
            
            # Создаем файл на Google Диске
            create_file_in_google_drive(message_text, parsed_data.get('date').isoformat())
            
            # Обновляем Google Sheet
            update_habits_google_sheet(parsed_data)
            update_goals_google_sheet(parsed_data)
            
            # Выявляем задачи из заметок по целям
            tasks = extract_tasks_from_texts(parsed_data)
            #logging.debug(f"Выявленные задачи: {tasks}")
            #tasks_message = format_tasks_for_message(tasks)
            
            # Создаем задачи в Todoist
            create_tasks_in_todoist(tasks)
            
            # Отправляем подтверждение в Telegram
            confirmation_message = "Данные успешно обработаны и сохранены."
            #full_message = f"{confirmation_message}\n\n{tasks_message}"
            send_telegram_message(chat_id, confirmation_message)
        else:
            # Обрабатываем ошибку парсинга
            error_message = "Не удалось извлечь необходимые данные из сообщения."
            logging.error(error_message)
            send_telegram_message(chat_id, error_message)

    except Exception as e:
        logging.error(f"Ошибка обработки сообщения: {e}")

    return "OK", 200
    
def create_tasks_in_todoist(tasks):
    todoist_token = os.getenv("TODOIST_API_TOKEN")
    headers = {
        "Authorization": f"Bearer {todoist_token}",
        "Content-Type": "application/json"
    }

    # Используем соответствие целей и проектов из конфигурации
    goal_to_project_mapping = config.get('goal_to_project_mapping', {})

    # Фильтруем задачи
    tasks_to_create = [task for task in tasks if task['task'] != 'Нет задач.']

    if not tasks_to_create:
        logging.info("Нет задач для создания в Todoist.")
        return

    for task in tasks_to_create:
        goal_name = task.get('goal', 'Общая заметка')
        task_description = task['task']

        # Получаем ID проекта на основе цели
        project_id = goal_to_project_mapping.get(goal_name)
        if not project_id:
            logging.warning(f"Для цели '{goal_name}' не найден проект. Используем проект по умолчанию.")
            project_id = goal_to_project_mapping.get('Общая заметка')  # Убедитесь, что этот ключ есть в конфиге

        # Формируем данные для задачи
        task_data = {
            "content": task_description,
            "project_id": project_id,
            "labels": ["journal bot"]
        }

        # Отправляем запрос на создание задачи
        try:
            response = requests.post(
                url="https://api.todoist.com/rest/v2/tasks",
                headers=headers,
                json=task_data
            )

            if response.status_code in (200, 201):
                logging.info(f"Задача '{task_description}' успешно создана в проекте с ID {project_id}.")
            else:
                logging.error(f"Не удалось создать задачу '{task_description}' в проекте с ID {project_id}: {response.status_code}, {response.text}")

        except Exception as e:
            logging.error(f"Ошибка при создании задачи в Todoist: {e}")


def format_tasks_for_message(tasks):
    # Создаем словарь для группировки задач по целям
    grouped_tasks = {}
    
    for task in tasks:
        goal = task['goal']
        task_description = task['task']
        
        # Если цели еще нет в словаре, создаем список для нее
        if goal not in grouped_tasks:
            grouped_tasks[goal] = []
        
        # Добавляем задачу в соответствующую цель
        grouped_tasks[goal].append(task_description)
    
    # Формируем текст для отправки
    message_lines = []
    for goal, task_list in grouped_tasks.items():
        message_lines.append(f"{goal}:")
        for task in task_list:
            message_lines.append(f"  - {task}")
        message_lines.append("")  # Пустая строка для разделения целей

    # Соединяем все строки в один текст
    return "\n".join(message_lines).strip()
    
def extract_tasks_from_texts(parsed_data, llm_provider=None, model_name=None):
    llm_provider = llm_provider or os.getenv('LLM_PROVIDER', 'openai')
    model_name = model_name or os.getenv('LLM_MODEL_NAME', 'gpt-4o-mini')
    # Настройка LLM в зависимости от поставщика
    if llm_provider == 'openai':
        os.environ["OPENAI_API_KEY"] = os.getenv("LLM_API_KEY")
        llm = OpenAI()
    else:
        raise ValueError("Неизвестный поставщик LLM")

    tasks = []

    # Обработка заметок по целям
    goals = parsed_data.get('goals', {})
    for goal_name, goal_info in goals.items():
        goal_note = goal_info.get('note', '')
        if not goal_note.strip():
            continue  # Пропускаем, если заметка по цели пустая

        extracted_tasks = extract_tasks_from_text(goal_note, llm, model_name)
        for task in extracted_tasks:
            tasks.append({
                'goal': goal_name,
                'task': task
            })

    # Обработка 'Заметки за день'
    day_note = parsed_data.get('day_note', '')
    if day_note.strip():
        extracted_tasks = extract_tasks_from_text(day_note, llm, model_name)
        for task in extracted_tasks:
            tasks.append({
                'goal': 'Общая заметка',  # Или другое обозначение
                'task': task
            })

    return tasks

def extract_tasks_from_text(text, llm, model_name):
    # Шаблон запроса
    template="""
В заметке описан прогресс или события за сегодня. Посмотри ее и если там есть описание будущих задач, то извлеки их. Задач может и не быть, тогда верни сообщение "Нет задач". Если задачи будут выявлены, то верни их в виде списка. В ответе указывай только задачи без дополнительных пояснений.

Заметка:
{note}

Формат вывода ответа:
- ...
"""
    prompt = template.format(note=text)

    # Формирование запроса для ChatCompletion
    messages = [
        {"role": "system", "content": "Ты помощник, который извлекает задачи из заметок."},
        {"role": "user", "content": prompt}
    ]

    # Получение ответа от модели
    completion = llm.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.5
    )

    # Обработка ответа
    response_text = completion.choices[0].message.content
    extracted_tasks = [task.strip('-• ').strip() for task in response_text.split('\n') if task.strip()]
    return extracted_tasks

    
def get_google_credentials():
    try:
        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"),
            scopes=[
                'https://www.googleapis.com/auth/drive',
                'https://www.googleapis.com/auth/spreadsheets'
            ]
        )
        return creds
    except Exception as e:
        logging.error(f"Ошибка при загрузке учетных данных Google: {e}")
        return None

def create_file_in_google_drive(content, date):
    try:
        # Получение учетных данных через отдельную функцию
        creds = get_google_credentials()
        if not creds:
            raise Exception("Не удалось загрузить учетные данные Google.")
        
        # Создание сервиса Google Drive API
        service = build('drive', 'v3', credentials=creds)
        
        # Метаданные файла
        file_metadata = {
            'name': f'{date}.txt',
            'parents': [os.getenv("GOOGLE_DRIVE_FOLDER_ID")]
        }
        
        # Содержимое файла
        media = MediaInMemoryUpload(content.encode('utf-8'), mimetype='text/plain')
        
        # Создание файла
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        
        logging.info(f"Файл создан на Google Диске с ID: {file.get('id')}")
    except Exception as e:
        logging.error(f"Ошибка при создании файла на Google Диске: {e}")

def update_habits_google_sheet(parsed_data):
    try:
        # Получение учетных данных через отдельную функцию
        creds = get_google_credentials()
        if not creds:
            raise Exception("Не удалось загрузить учетные данные Google.")

        # Создание сервиса Google Sheets API
        service = build('sheets', 'v4', credentials=creds)

        # ID таблицы Google Sheets, куда вы хотите внести данные
        spreadsheet_id = os.getenv("GOOGLE_SHEET_ID")

        # Имя листа с привычками
        sheet_name = 'Привычки-неделя'

        # Список всех ваших привычек
        all_habits = ['Зарядка', 'Кегель', 'Стоицизм', 'Чтение', 'Сербский', 'Трекер работы', 'Английский', 'Запись в дневник']

        # Подготовка данных о выполненных привычках
        habits_checked = parsed_data.get('habits_checked', [])
        habit_values = ['✔' if habit in habits_checked else '' for habit in all_habits]

        # Получение текущих данных с листа, чтобы найти первую пустую строку
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f'{sheet_name}!A:A'
        ).execute()

        values = result.get('values', [])
        next_row = len(values) + 1  # Следующая доступная строка

        # Подготовка данных для записи
        date_str = parsed_data.get('date').strftime('%d.%m.%Y') if parsed_data.get('date') else ''
        values = [[date_str] + habit_values]

        # Диапазон для записи данных
        range_name = f'{sheet_name}!A{next_row}:I{next_row}'  # A - дата, B-I - привычки

        body = {
            'values': values
        }

        # Добавление данных в таблицу
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption='USER_ENTERED',
            body=body
        ).execute()

        logging.info(f"Данные по привычкам успешно добавлены в Google Sheet.")

    except Exception as e:
        logging.error(f"Ошибка при обновлении Google Sheet для привычек: {e}")

def update_goals_google_sheet(parsed_data):
    try:
        # Получение учетных данных через отдельную функцию
        creds = get_google_credentials()
        if not creds:
            raise Exception("Не удалось загрузить учетные данные Google.")

        # Создание сервиса Google Sheets API
        service = build('sheets', 'v4', credentials=creds)

        # ID таблицы Google Sheets, куда вы хотите внести данные
        spreadsheet_id = os.getenv("GOOGLE_SHEET_ID")

        # Имя листа с целями
        sheet_name = 'Прогресс по целям'

        # Список всех целей
        all_goals = [
            'Финансовая цель', 'Бизнес цель', 'Семейная цель', 
            'Цель по английскому', 'Цель по сербскому', 
            'Цель обучения PM', 'Цель по физической форме'
        ]

        # Подготовка данных по процентам выполнения целей
        goals = parsed_data.get('goals', {})
        goal_progress = [goals.get(goal, {}).get('completion_percentage', '') for goal in all_goals]

        # Получение текущих данных с листа, чтобы найти первую пустую строку
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f'{sheet_name}!A:A'
        ).execute()

        values = result.get('values', [])
        next_row = len(values) + 1  # Следующая доступная строка

        # Подготовка данных для записи
        date_str = parsed_data.get('date').strftime('%d.%m.%Y') if parsed_data.get('date') else ''
        values = [[date_str] + goal_progress]

        # Диапазон для записи данных
        range_name = f'{sheet_name}!A{next_row}:H{next_row}'  # A - дата, B-H - цели

        body = {
            'values': values
        }

        # Добавление данных в таблицу
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption='USER_ENTERED',
            body=body
        ).execute()

        logging.info(f"Данные по целям успешно добавлены в Google Sheet.")

    except Exception as e:
        logging.error(f"Ошибка при обновлении Google Sheet для целей: {e}")


# Функция отправки ответа пользователю
def send_telegram_message(chat_id, text):
    try:
        url = f"{URL}sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text
        }
        response = requests.post(url, json=payload)
        logging.info(f"Ответ на отправку сообщения: {response.status_code} - {response.text}")

    except Exception as e:
        logging.error(f"Ошибка при отправке сообщения: {e}")
        
def save_data_to_database(data):
    conn = sqlite3.connect('journalbot_database.db')
    cursor = conn.cursor()
    # Обновляем схему таблицы
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parsed_data (
            date TEXT,
            my_mood TEXT,
            wife_mood TEXT,
            habits_checked TEXT,
            goals TEXT,
            day_note TEXT,
            events_emojis TEXT,
            self_care_emojis TEXT,
            sleep_start TEXT,
            sleep_end TEXT
        )
    ''')
    # Вставляем данные
    cursor.execute('''
        INSERT INTO parsed_data (
            date, my_mood, wife_mood, habits_checked, goals, day_note, events_emojis, self_care_emojis, sleep_start, sleep_end
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('date').isoformat() if data.get('date') else None,
        data.get('my_mood'),
        data.get('wife_mood'),
        json.dumps(data.get('habits_checked', [])),
        json.dumps(data.get('goals')),
        data.get('day_note'),
        data.get('events_emojis'),
        data.get('self_care_emojis'),
        data.get('sleep_duration', {}).get('start'),
        data.get('sleep_duration', {}).get('end')
    ))
    conn.commit()
    conn.close()


def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # Эмоции
                           u"\U0001F300-\U0001F5FF"  # Символы и пиктограммы
                           u"\U0001F680-\U0001F6FF"  # Транспорт и символы карты
                           u"\U0001F1E0-\U0001F1FF"  # Флаги
                           u"\U00002700-\U000027BF"  # Дингбаты
                           u"\U0001F900-\U0001F9FF"  # Дополнительные символы и пиктограммы
                           u"\U00002600-\U000026FF"  # Разные символы
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text).strip()

def parse_telegram_message(message):
    data = {}
    success = False
    try:
        logging.debug(f"Сообщение для парсинга: {message}")
        
        patterns = config['parsing_patterns']

        # Извлечение даты
        date_match = re.search(patterns.get('date', ''), message)
        if date_match:
            data['date'] = datetime.strptime(date_match.group(1), '%d.%m.%Y').date()
        else:
            data['date'] = None

        # Извлечение настроения
        mood_match = re.search(patterns.get('my_mood', ''), message)
        if mood_match:
            data['my_mood'] = mood_match.group(1).strip()
        else:
            data['my_mood'] = None

        # Извлечение настроения жены
        wife_mood_match = re.search(patterns.get('wife_mood', ''), message)
        if wife_mood_match:
            data['wife_mood'] = wife_mood_match.group(1).strip()
        else:
            data['wife_mood'] = None
        
        health_match = re.search(patterns.get('health', ''), message)
        if health_match:
            data['health'] = health_match.group(1).strip()
        else:
            data['health'] = None
        
        # Извлечение привычек с отметкой ✅
        habits_section = re.search(patterns.get('habits_section', ''), message)
        if habits_section:
            habits_text = habits_section.group(1)
            habit_lines = habits_text.strip().split('\n')
            habits_checked = []
            for habit_line in habit_lines:
                if '✅' in habit_line:
                    habit_match = re.match(patterns.get('habit_checked', ''), habit_line.strip())
                    if habit_match:
                        habit_name = habit_match.group(1).strip()
                        habits_checked.append(habit_name)
            data['habits_checked'] = habits_checked
        else:
            data['habits_checked'] = []

        # Извлечение целей и их процентов выполнения
        goals_section = re.search(patterns.get('goals_section', ''), message)
        if goals_section:
            goals_text = goals_section.group(1)
            goal_blocks = re.findall(patterns.get('goal_blocks', ''), goals_text, re.DOTALL)
            goals = {}
            for goal in goal_blocks:
                goal_name = goal[0].strip()
                goal_note = goal[1].strip()
                goal_percent = float(goal[2].replace(',', '.'))
                goals[goal_name] = {
                    'note': goal_note,
                    'completion_percentage': goal_percent
                }
            data['goals'] = goals
        else:
            data['goals'] = {}

        # Извлечение 'Заметки за день'
        day_note_match = re.search(patterns.get('day_note_section', ''), message)
        if day_note_match:
            data['day_note'] = day_note_match.group(1).strip()
        else:
            data['day_note'] = ''

        # Извлечение 'Уход за собой'
        self_care_match = re.search(patterns.get('self_care_section', ''), message)
        if self_care_match:
            data['self_care_emojis'] = self_care_match.group(1).strip()
        else:
            data['self_care_emojis'] = ''

        # Извлечение эмодзи событий
        events_match = re.search(patterns.get('events', ''), message)
        if events_match:
            data['events_emojis'] = events_match.group(1).strip()
        else:
            data['events_emojis'] = ''

        # Извлечение продолжительности сна
        sleep_match = re.search(patterns.get('sleep', ''), message)
        if sleep_match:
            start_sleep = sleep_match.group(1)
            end_sleep = sleep_match.group(2)
            data['sleep_duration'] = {
                'start': start_sleep,
                'end': end_sleep
            }
        else:
            data['sleep_duration'] = {} 
        
        # Check if critical fields are parsed successfully
        if data.get('date') and data.get('my_mood'):
            success = True
        else:
            success = False
    except Exception as e:
        logging.error(f"Ошибка при парсинге сообщения: {e}")
        success = False
        
    return data, success




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
