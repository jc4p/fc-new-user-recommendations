#!/usr/bin/env python3
"""
Interactive CLI for personalized content recommendations using binary search
"""
import os
import sys
import json
from typing import List, Dict
import textwrap
from recommendation_utils import ClusterRecommender

# ANSI color codes for better CLI experience
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print welcome header"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 70)
    print("       FARCASTER CONTENT RECOMMENDATION SYSTEM")
    print("=" * 70)
    print(f"{Colors.END}")

def print_post(post: Dict, question_num: int, total_questions: int):
    """Display a post nicely formatted"""
    print(f"\n{Colors.YELLOW}Post {question_num}/{total_questions}{Colors.END}")
    print("-" * 70)
    
    # Wrap text nicely
    wrapped_text = textwrap.fill(post['text'], width=70)
    print(f"{Colors.CYAN}{wrapped_text}{Colors.END}")
    
    # Show metadata
    print("-" * 70)
    print(f"{Colors.GREEN}Author Quality: {post['author_quality']:.1f}/10")
    print(f"Quality Reactions: {post['quality_reactions']} | Quality Replies: {post['quality_replies']}{Colors.END}")
    print("-" * 70)

def get_user_response() -> bool:
    """Get user's like/dislike response"""
    while True:
        response = input(f"\n{Colors.BOLD}Do you like this post? (y/n):{Colors.END} ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print(f"{Colors.RED}Please enter 'y' for yes or 'n' for no.{Colors.END}")

def print_preference_summary(summary: Dict):
    """Print the user's learned preferences"""
    print(f"\n{Colors.BOLD}{Colors.GREEN}Based on your choices, you seem to enjoy:{Colors.END}")
    
    if summary['preferred_themes']:
        print(f"\n{Colors.YELLOW}Themes:{Colors.END}")
        for theme in summary['preferred_themes']:
            print(f"  • {theme.title()}")
    
    if summary['top_keywords']:
        print(f"\n{Colors.YELLOW}Topics containing:{Colors.END}")
        keywords_str = ", ".join(summary['top_keywords'][:6])
        print(f"  {keywords_str}")

def print_recommendations(recommendations: List[Dict]):
    """Print the final recommendations"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}")
    print("YOUR PERSONALIZED RECOMMENDATIONS")
    print('=' * 70 + f"{Colors.END}\n")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{Colors.BOLD}{i}. [{rec['cluster_theme'].upper()}]{Colors.END}")
        
        # Truncate text for display
        display_text = rec['text'][:200] + "..." if len(rec['text']) > 200 else rec['text']
        wrapped = textwrap.fill(display_text, width=70, initial_indent="   ", subsequent_indent="   ")
        print(wrapped)
        
        print(f"   {Colors.GREEN}Quality: {rec['author_quality']:.1f}/10 | "
              f"Reactions: {rec['quality_reactions']} | "
              f"Replies: {rec['quality_replies']}{Colors.END}")
        
        # print(f"   {Colors.CYAN}Hash: {rec['hash'][:16]}...{Colors.END}")
        print()

def print_recommended_users(users: List[Dict]):
    """Print recommended users to follow"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}")
    print("RECOMMENDED USERS TO FOLLOW")
    print('=' * 70 + f"{Colors.END}\n")
    
    for i, user in enumerate(users, 1):
        print(f"{Colors.BOLD}{i}. FID: {user['fid']}{Colors.END}")
        print(f"   {Colors.GREEN}Quality Score: {user['quality_score']:.1f}/10{Colors.END}")
        print(f"   {Colors.YELLOW}Themes: {', '.join(user['themes'])}{Colors.END}")
        if user['sample_post']:
            print(f"   {Colors.CYAN}Sample: {user['sample_post']}...{Colors.END}")
        print()

def save_recommendations(recommendations: List[Dict], preferences: Dict, recommended_users: List[Dict] = None):
    """Save recommendations to file"""
    output = {
        'preferences': preferences,
        'recommendations': recommendations,
        'recommended_users': recommended_users if recommended_users else []
    }
    
    filename = 'recommendations_output.json'
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"{Colors.GREEN}Recommendations saved to {filename}{Colors.END}")

def main():
    """Main CLI interaction loop"""
    clear_screen()
    print_header()
    
    print(f"{Colors.CYAN}Welcome! I'll help you discover great content on Farcaster.")
    print(f"I'll show you up to 5 posts and learn what you like.{Colors.END}\n")
    
    input(f"{Colors.BOLD}Press Enter to begin...{Colors.END}")
    
    # Check if clusters file exists
    if not os.path.exists('data/quality_clusters.json'):
        print(f"{Colors.RED}Error: Cluster data not found!")
        print(f"Please run 'python cluster_quality_content.py' first.{Colors.END}")
        return
    
    # Initialize recommender
    try:
        recommender = ClusterRecommender()
    except Exception as e:
        print(f"{Colors.RED}Error loading cluster data: {e}{Colors.END}")
        return
    
    # Binary search loop
    max_questions = 7
    
    for question_num in range(1, max_questions + 1):
        clear_screen()
        print_header()
        
        # Get next post
        try:
            post, cluster_id = recommender.select_next_post(question_num - 1)
        except Exception as e:
            print(f"{Colors.RED}Error selecting post: {e}{Colors.END}")
            break
        
        # Show post
        print_post(post, question_num, max_questions)
        
        # Get response
        liked = get_user_response()
        
        # Update weights
        recommender.update_weights(cluster_id, liked)
        
        # Optional: stop early if we have strong signal
        if question_num >= 4:
            # Check if we have a strong preference signal
            weights = list(recommender.cluster_weights.values())
            max_weight = max(weights)
            avg_weight = sum(weights) / len(weights)
            
            if max_weight > avg_weight * 3:  # Strong preference emerged
                print(f"\n{Colors.GREEN}I think I understand your preferences!{Colors.END}")
                response = input("Would you like to see recommendations now? (y/n): ").lower()
                if response in ['y', 'yes']:
                    break
    
    # Generate recommendations
    clear_screen()
    print_header()
    
    print(f"{Colors.YELLOW}Analyzing your preferences...{Colors.END}\n")
    
    # Get preference summary
    preferences = recommender.get_preference_summary()
    print_preference_summary(preferences)
    
    # Get recommendations
    recommendations = recommender.get_recommendations(n=10)
    print_recommendations(recommendations)
    
    # Get recommended users
    recommended_users = recommender.get_recommended_users(n=10)
    print_recommended_users(recommended_users)
    
    # Offer to save
    save_response = input(f"\n{Colors.BOLD}Save these recommendations? (y/n):{Colors.END} ").lower()
    if save_response in ['y', 'yes']:
        save_recommendations(recommendations, preferences, recommended_users)
    
    print(f"\n{Colors.CYAN}Thank you for using the recommendation system!")
    print(f"Run this again anytime to get fresh recommendations.{Colors.END}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Recommendation session cancelled.{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}An error occurred: {e}{Colors.END}")
        sys.exit(1)
