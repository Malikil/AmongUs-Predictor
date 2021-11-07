Among Us Predictor
===

The goal of this project is to come up with a model which will be able to predict whether crewmates or impostors should have won in a game of Among Us. It's not intended to be used predictively before a match, as at that time the roles of players aren't known. The idea is to eventually use it in the construction of a rating system, so the user would be able to generate a leaderboard of the players in their group for whatever roles or game settings.

---

## Data Format

Format input data as follows  
```js
[
    {
        crewmates: [
            "List",
            "Of",
            "Crewmates"
        ],
        impostors: [
            "Impostor",
            "List"
        ],
        map: "Airship",                 // "The Skeld" | "MIRA HQ" | "Polus" | "Airship"
        confirm_ejects: false,
        emergency_meetings: 1,
        emergency_cooldown: 20,
        discussion_time: 15,
        voting_time: 120,
        anonymous_votes: true,
        player_speed: 1.25,
        crewmate_vision: 0.75,
        impostor_vision: 1.5,
        kill_cooldown: 35.0,
        kill_distance: "Short",         // "Short" | "Medium" | "Long"
        visual_tasks: false,
        task_bar_updates: "Meetings",   // "Always" | "Meetings" | "Never"
        common_tasks: 2,
        long_tasks: 3,
        short_tasks: 5,
        match_winner: "crewmates"       // "crewmates" | "impostors"
    },
    { ... }
]
```