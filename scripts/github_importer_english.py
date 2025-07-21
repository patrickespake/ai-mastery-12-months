#!/usr/bin/env python3
"""
GitHub Project Importer
Python script to automatically import milestones and issues to GitHub Projects
"""

import requests
import csv
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

class GitHubProjectImporter:
    def __init__(self, username: str, repo: str, token: str):
        self.username = username
        self.repo = repo
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        }
        
    def create_milestone(self, title: str, description: str, due_date: str) -> Optional[Dict]:
        """Create milestone in repository"""
        url = f"{self.base_url}/repos/{self.username}/{self.repo}/milestones"
        
        data = {
            "title": title,
            "description": description,
            "due_on": due_date,
            "state": "open"
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 201:
                print(f"✅ Milestone created: {title}")
                return response.json()
            elif response.status_code == 422:
                print(f"⚠️  Milestone already exists: {title}")
                return None
            else:
                print(f"❌ Error creating milestone {title}: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return None
    
    def get_milestone_number(self, title: str) -> Optional[int]:
        """Get milestone number by title"""
        url = f"{self.base_url}/repos/{self.username}/{self.repo}/milestones"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                milestones = response.json()
                for milestone in milestones:
                    if milestone['title'] == title:
                        return milestone['number']
            return None
        except Exception as e:
            print(f"❌ Error fetching milestones: {e}")
            return None
    
    def create_issue(self, title: str, body: str, milestone: str, labels: List[str]) -> Optional[Dict]:
        """Create issue in repository"""
        url = f"{self.base_url}/repos/{self.username}/{self.repo}/issues"
        
        # Get milestone number
        milestone_number = self.get_milestone_number(milestone)
        
        data = {
            "title": title,
            "body": body,
            "labels": labels,
        }
        
        if milestone_number:
            data["milestone"] = milestone_number
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 201:
                print(f"✅ Issue created: {title}")
                return response.json()
            else:
                print(f"❌ Error creating issue {title}: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return None
    
    def create_label(self, name: str, color: str, description: str) -> bool:
        """Create label in repository"""
        url = f"{self.base_url}/repos/{self.username}/{self.repo}/labels"
        
        data = {
            "name": name,
            "color": color,
            "description": description
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 201:
                print(f"✅ Label created: {name}")
                return True
            elif response.status_code == 422:
                print(f"⚠️  Label already exists: {name}")
                return True
            else:
                print(f"❌ Error creating label {name}: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def import_from_csv(self, milestones_file: str, issues_file: str):
        """Import milestones and issues from CSV files"""
        
        print("🚀 Starting import...")
        
        # Import milestones
        print("\n📅 Importing milestones...")
        try:
            with open(milestones_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.create_milestone(
                        title=row['title'],
                        description=row['description'],
                        due_date=row['due_date'] + "T00:00:00Z"
                    )
                    time.sleep(1)  # Rate limiting
        except FileNotFoundError:
            print(f"❌ File not found: {milestones_file}")
            return
        
        # Wait for milestones to be processed
        print("\n⏳ Waiting for milestones processing...")
        time.sleep(5)
        
        # Import issues
        print("\n📝 Importing issues...")
        try:
            with open(issues_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    labels = [label.strip() for label in row['labels'].split(',')]
                    
                    self.create_issue(
                        title=row['title'],
                        body=row['body'],
                        milestone=row['milestone'],
                        labels=labels
                    )
                    
                    # Rate limiting - GitHub allows 5000 requests/hour
                    if i % 10 == 0:
                        print(f"⏳ Processed {i+1} issues...")
                        time.sleep(2)
                    else:
                        time.sleep(0.5)
                        
        except FileNotFoundError:
            print(f"❌ File not found: {issues_file}")
            return
        
        print("\n✅ Import completed!")
    
    def setup_labels(self):
        """Setup default project labels"""
        print("\n🏷️  Setting up labels...")
        
        labels = [
            # Activity type
            {"name": "theory", "color": "0052cc", "description": "Theoretical and conceptual activities"},
            {"name": "practice", "color": "5319e7", "description": "Practical exercises and implementations"},
            {"name": "project", "color": "d73a4a", "description": "Complete and integrative projects"},
            {"name": "networking", "color": "0e8a16", "description": "Networking and community activities"},
            
            # Knowledge area
            {"name": "mathematics", "color": "f9d0c4", "description": "Mathematics and statistics"},
            {"name": "programming", "color": "c2e0c6", "description": "Programming and development"},
            {"name": "machine-learning", "color": "bfd4f2", "description": "Classical Machine Learning"},
            {"name": "deep-learning", "color": "d1ecf1", "description": "Deep Learning and neural networks"},
            {"name": "ai-generative", "color": "fef2c0", "description": "Generative AI and advanced models"},
            
            # Priority
            {"name": "priority-high", "color": "d73a4a", "description": "High priority"},
            {"name": "priority-medium", "color": "fbca04", "description": "Medium priority"},
            {"name": "priority-low", "color": "0052cc", "description": "Low priority"},
            
            # Setup
            {"name": "setup", "color": "7057ff", "description": "Configuration and setup"},
            {"name": "environment", "color": "008672", "description": "Development environment"},
        ]
        
        # Add week labels (1-48)
        for week in range(1, 49):
            labels.append({
                "name": f"week-{week}",
                "color": "e4e669",
                "description": f"Week {week}"
            })
        
        for label in labels:
            self.create_label(label["name"], label["color"], label["description"])
            time.sleep(0.2)  # Rate limiting

def main():
    """Main function"""
    print("🎯 GitHub Project Importer for AI Study Plan")
    print("=" * 50)
    
    # Configuration (EDIT HERE)
    USERNAME = input("Enter your GitHub username: ").strip()
    REPO = input("Enter repository name: ").strip()
    TOKEN = input("Enter your GitHub token: ").strip()
    
    if not all([USERNAME, REPO, TOKEN]):
        print("❌ All fields are required!")
        return
    
    # Create importer
    importer = GitHubProjectImporter(USERNAME, REPO, TOKEN)
    
    # Options menu
    while True:
        print("\n📋 Available options:")
        print("1. Setup labels")
        print("2. Import milestones and issues from CSV")
        print("3. Create milestones only")
        print("4. Exit")
        
        choice = input("\nChoose an option (1-4): ").strip()
        
        if choice == "1":
            importer.setup_labels()
            
        elif choice == "2":
            milestones_file = input("Milestones file path (default: milestones_import_english.csv): ").strip()
            if not milestones_file:
                milestones_file = "milestones_import_english.csv"
                
            issues_file = input("Issues file path (default: all_issues_complete_english.csv): ").strip()
            if not issues_file:
                issues_file = "all_issues_complete_english.csv"
                
            importer.import_from_csv(milestones_file, issues_file)
            
        elif choice == "3":
            # Create milestones manually
            milestones = [
                {
                    "title": "MILESTONE 1: Fundamentals and Mathematics",
                    "description": "Establish a solid foundation in mathematics, Python programming, and fundamental AI concepts",
                    "due_date": "2025-08-15"
                },
                {
                    "title": "MILESTONE 2: Classical Machine Learning",
                    "description": "Master traditional Machine Learning algorithms and validation techniques",
                    "due_date": "2025-10-15"
                },
                {
                    "title": "MILESTONE 3: Deep Learning and Neural Networks",
                    "description": "Specialize in neural networks, CNNs, RNNs and advanced architectures",
                    "due_date": "2025-12-15"
                },
                {
                    "title": "MILESTONE 4: Generative AI and Advanced Models",
                    "description": "Master generative AI, GANs, VAEs, Diffusion Models and LLMs",
                    "due_date": "2026-02-15"
                },
                {
                    "title": "MILESTONE 5: Specialization and Applications",
                    "description": "Apply knowledge in specific domains and prepare for production",
                    "due_date": "2026-04-15"
                },
                {
                    "title": "MILESTONE 6: Advanced Projects and Portfolio",
                    "description": "Create professional portfolio and complex projects to demonstrate expertise",
                    "due_date": "2026-06-15"
                }
            ]
            
            for milestone in milestones:
                importer.create_milestone(
                    milestone["title"],
                    milestone["description"],
                    milestone["due_date"]
                )
                time.sleep(1)
                
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option!")

if __name__ == "__main__":
    main()

