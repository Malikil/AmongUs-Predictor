from flask import Flask
from flask import request
from flask import render_template
from flask import redirect

app = Flask(__name__)

def validate_data(form_data):
    data = {}
    keys = [
        ('player_count', int),
        ('impostor_count', int),
        ('map', str),
        ('confirm_ejects', bool),
        ('emergency_meetings', int),
        ('emergency_cooldown', int),
        ('discussion_time', int),
        ('voting_time', int),
        ('anonymous_votes', bool),
        ('player_speed', float),
        ('crewmate_vision', float),
        ('impostor_vision', float),
        ('kill_cooldown', float),
        ('kill_distance', str),
        ('visual_tasks', bool),
        ('task_bar_updates', str),
        ('common_tasks', int),
        ('long_tasks', int),
        ('short_tasks', int)
    ]
    for k in keys:
        data[k[0]] = k[1](form_data.get(k[0]))
    return data

@app.route('/')
def root():
    return render_template('leaders-form.html')

@app.route('/leaders', methods=['GET', 'POST'])
def leaders():
    try:
        jdata = validate_data(request.form)
    except TypeError:
        return redirect('/')
        
    # Get some leaders
    print(jdata)

    return {
        'settings': jdata
    }

if __name__ == '__main__':
    app.run(debug=True)
