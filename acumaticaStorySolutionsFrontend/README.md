# Story Solutions Frontend

A React-based frontend application for processing JIRA stories and generating comprehensive markdown solutions using AI-powered RAG (Retrieval-Augmented Generation).

## Features

- 📝 **Story Processing Form**: Input JIRA story details including description and acceptance criteria
- 🤖 **AI-Powered Solutions**: Generate comprehensive solutions using RAG technology
- 📄 **Markdown Rendering**: Beautiful markdown display with syntax highlighting
- 📚 **Source References**: View document sources used in solution generation
- 💾 **Export Solutions**: Copy or download solutions as markdown files
- 🔍 **Health Monitoring**: Real-time backend service health status

## Tech Stack

- **React 19** - UI library
- **Vite** - Build tool and dev server
- **Material-UI (MUI)** - Component library
- **Axios** - HTTP client
- **React Markdown** - Markdown rendering
- **React Syntax Highlighter** - Code syntax highlighting
- **React Hot Toast** - Toast notifications

## Getting Started

### Prerequisites

- Node.js 20.18.3 or higher
- npm or yarn
- Backend service running on port 8001

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open your browser and navigate to `http://localhost:5173`

### Building for Production

```bash
npm run build
```

The production build will be in the `dist` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
src/
├── api/                 # API configuration and services
│   ├── config.js       # API configuration
│   └── solutions.js    # Solutions API service
├── components/          # React components
│   ├── HealthStatus.jsx
│   ├── LoadingSpinner.jsx
│   ├── SolutionDisplay.jsx
│   ├── SourceReferences.jsx
│   └── StoryForm.jsx
├── App.jsx             # Main application component
├── main.jsx            # Application entry point
├── theme.js            # Material-UI theme configuration
└── index.css           # Global styles
```

## API Configuration

The frontend connects to the backend API running on port 8001. You can modify the API base URL in `src/api/config.js`.

## Usage

1. **Enter Story Details**:
   - Story ID (optional)
   - Title (optional)
   - Description (required)
   - Acceptance Criteria (required, at least one)

2. **Generate Solution**:
   - Click "Generate Solution" button
   - Wait for processing (may take 30 seconds to 5 minutes)

3. **View Results**:
   - Review the generated markdown solution
   - Check source references used
   - Copy or download the solution

## Development

### Code Style

- Use `.jsx` extension for React components
- Follow React best practices and hooks patterns
- Use Material-UI components for consistent UI
- Implement proper error handling and loading states

### API Integration

All API calls are abstracted in `src/api/solutions.js`:
- `processStory(storyData)` - Process a JIRA story
- `healthCheck()` - Check service health
- `apiHealthCheck()` - Check general API health

## License

Private project - All rights reserved
