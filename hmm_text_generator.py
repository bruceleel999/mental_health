import numpy as np
from collections import defaultdict

class HMMTextGenerator:
    def __init__(self):
        # Define states (stress levels and factors)
        self.states = ['Low', 'Moderate', 'High']
        self.factors = ['Sleep_Quality', 'Physical_Activity', 'Social_Support']
        
        # Initialize transition probabilities
        self.transition_probs = {
            'Low': {'Low': 0.6, 'Moderate': 0.3, 'High': 0.1},
            'Moderate': {'Low': 0.3, 'Moderate': 0.4, 'High': 0.3},
            'High': {'Low': 0.1, 'Moderate': 0.3, 'High': 0.6}
        }
        
        # Define observation sequences (words/phrases)
        self.observations = {
            'Low': [
                "You're managing stress well",
                "Great job maintaining balance",
                "Your approach is working effectively",
                "Keep up the good work",
                "You're on the right track"
            ],
            'Moderate': [
                "Some stress is manageable",
                "Room for improvement",
                "You're handling things okay",
                "Could use some adjustments",
                "Making progress"
            ],
            'High': [
                "Significant stress detected",
                "Important to address this",
                "Consider seeking help",
                "Time to take action",
                "Focus on reducing stress"
            ]
        }
        
        # Define emission probabilities for each state
        self.emission_probs = {
            'Low': {word: 1.0/len(self.observations['Low']) for word in self.observations['Low']},
            'Moderate': {word: 1.0/len(self.observations['Moderate']) for word in self.observations['Moderate']},
            'High': {word: 1.0/len(self.observations['High']) for word in self.observations['High']}
        }
        
        # Factor-specific recommendations
        self.factor_recommendations = {
            'Sleep_Quality': {
                'Poor': [
                    "Establish a regular sleep schedule",
                    "Create a relaxing bedtime routine",
                    "Limit screen time before bed"
                ],
                'Average': [
                    "Maintain consistent sleep patterns",
                    "Consider your sleep environment",
                    "Try to improve sleep quality"
                ],
                'Good': [
                    "Keep up your healthy sleep routine",
                    "Maintain your current sleep habits",
                    "Your sleep quality is excellent"
                ]
            },
            'Physical_Activity': {
                'Low': [
                    "Incorporate more physical activity",
                    "Start with short walks",
                    "Try light exercise"
                ],
                'Moderate': [
                    "Maintain your activity level",
                    "Add variety to your routine",
                    "Keep up the good work"
                ],
                'High': [
                    "Excellent activity level",
                    "Continue your exercise habits",
                    "Your activity is ideal"
                ]
            },
            'Social_Support': {
                'Low': [
                    "Reach out to friends more often",
                    "Build stronger connections",
                    "Join a club or group"
                ],
                'Moderate': [
                    "Maintain current relationships",
                    "Strengthen social connections",
                    "Expand your social circle"
                ],
                'High': [
                    "Great social support system",
                    "Keep nurturing relationships",
                    "Your connections are strong"
                ]
            }
        }

    def viterbi(self, observations):
        """Viterbi algorithm for finding the most likely state sequence."""
        V = [{}]
        path = {}
        
        # Initialize base cases
        for state in self.states:
            V[0][state] = self.transition_probs['Low'][state] * self.emission_probs[state].get(observations[0], 1e-10)
            path[state] = [state]
        
        # Run Viterbi for t > 0
        for t in range(1, len(observations)):
            V.append({})
            newpath = {}
            
            for state in self.states:
                (prob, state_path) = max(
                    (V[t-1][prev_state] * self.transition_probs[prev_state][state] * 
                     self.emission_probs[state].get(observations[t], 1e-10), prev_state)
                    for prev_state in self.states
                )
                V[t][state] = prob
                newpath[state] = path[state_path] + [state]
            
            path = newpath
        
        # Find the most likely sequence
        (prob, state) = max((V[len(observations)-1][state], state) for state in self.states)
        return (prob, path[state])

    def generate_response(self, stress_level, factors):
        """Generate a personalized response using HMM."""
        # Generate main message based on stress level
        main_message = np.random.choice(self.observations[stress_level])
        
        # Generate factor-specific recommendations
        recommendations = []
        for factor, value in factors.items():
            if factor in self.factor_recommendations and value in self.factor_recommendations[factor]:
                rec = np.random.choice(self.factor_recommendations[factor][value])
                recommendations.append(rec)
        
        # Combine into final response
        response = f"{main_message}\n\n"
        response += "Recommendations:\n"
        for rec in recommendations:
            response += f"- {rec}\n"
        
        # Add closing message based on stress level
        if stress_level == "Low":
            response += "\nKeep up the good work and continue practicing these healthy habits!"
        elif stress_level == "Moderate":
            response += "\nWith some small changes, you can improve your stress management."
        else:
            response += "\nIt's important to take action to reduce your stress levels. Consider seeking professional help if needed."
        
        return response 