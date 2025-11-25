#!/bin/bash

# Quick Setup Script for Multi-Actor System
# This script helps set up the actors directory structure

echo "=========================================="
echo "  Lip-Sync Pipeline - Actor Setup"
echo "=========================================="
echo ""

# Check if actors directory exists
if [ ! -d "actors" ]; then
    echo "Creating actors directory..."
    mkdir -p actors
fi

# Check for actors_config.yaml
if [ ! -f "actors/actors_config.yaml" ]; then
    echo "Creating default actors_config.yaml..."
    cat > actors/actors_config.yaml << 'EOF'
# Central configuration for all actors
# This file is optional - individual metadata.yaml files can override these settings

default_voice:
  backend: gtts
  params:
    lang: en
    tld: com
    slow: false

actors:
  actor_1:
    display_name: "Professional Voice"
    description: "Clear, professional voice with neutral accent"
    voice:
      backend: gtts
      params:
        lang: en
        tld: com
        slow: false
  
  actor_2:
    display_name: "British Voice"
    description: "British English accent"
    voice:
      backend: gtts
      params:
        lang: en
        tld: co.uk
        slow: false
  
  actor_3:
    display_name: "Educational Voice"
    description: "Slower, clearer speech for educational content"
    voice:
      backend: gtts
      params:
        lang: en
        tld: com
        slow: true
EOF
    echo "✓ Created actors_config.yaml"
else
    echo "✓ actors_config.yaml already exists"
fi

# Function to create actor directory
create_actor() {
    local actor_name=$1
    local display_name=$2
    local description=$3
    local tld=${4:-com}
    local slow=${5:-false}
    
    echo ""
    echo "Setting up $actor_name..."
    
    if [ ! -d "actors/$actor_name" ]; then
        mkdir -p "actors/$actor_name/visemes_library"
        echo "  ✓ Created directory structure"
    else
        if [ ! -d "actors/$actor_name/visemes_library" ]; then
            mkdir -p "actors/$actor_name/visemes_library"
            echo "  ✓ Created visemes_library directory"
        else
            echo "  ✓ Directory already exists"
        fi
    fi
    
    if [ ! -f "actors/$actor_name/metadata.yaml" ]; then
        cat > "actors/$actor_name/metadata.yaml" << EOF
# Actor-specific metadata (overrides central config)

name: $actor_name
display_name: "$display_name"
description: "$description"

voice:
  backend: gtts
  params:
    lang: en
    tld: $tld
    slow: $slow

# Optional: additional metadata
tags:
  - custom

# Optional: supported languages
supported_languages:
  - en
EOF
        echo "  ✓ Created metadata.yaml"
    else
        echo "  ✓ metadata.yaml already exists"
    fi
}

# Check if we should create example actors
read -p "Do you want to create example actor directories? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    create_actor "actor_1" "Professional Voice" "Clear, professional voice with neutral accent" "com" "false"
    create_actor "actor_2" "British Voice" "British English accent" "co.uk" "false"
    create_actor "actor_3" "Educational Voice" "Slower, clearer speech for educational content" "com" "true"
    
    echo ""
    echo "=========================================="
    echo "⚠️  IMPORTANT: Add Viseme Libraries"
    echo "=========================================="
    echo ""
    echo "You need to add viseme images to each actor's visemes_library folder:"
    echo ""
    echo "  actors/actor_1/visemes_library/  <- Add viseme PNG/JPG files here"
    echo "  actors/actor_2/visemes_library/  <- Add viseme PNG/JPG files here"
    echo "  actors/actor_3/visemes_library/  <- Add viseme PNG/JPG files here"
    echo ""
    echo "Without viseme images, actors will appear in the list but"
    echo "video generation will be disabled with a warning message."
    echo ""
fi

# Check if PyYAML is installed
echo ""
echo "Checking Python dependencies..."
python3 -c "import yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  PyYAML is not installed!"
    read -p "Install PyYAML now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install pyyaml
        echo "✓ PyYAML installed"
    else
        echo "⚠️  Please install PyYAML manually: pip install pyyaml"
    fi
else
    echo "✓ PyYAML is installed"
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Add viseme images to actors/*/visemes_library/ folders"
echo "2. Customize voice settings in metadata.yaml files"
echo "3. Run the Streamlit app: streamlit run streamlit_app.py"
echo ""
echo "For more information, see actors/README.md"
echo ""
