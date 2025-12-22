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

   **For local access only:**
   ```bash
   npm run dev
   ```
   Then open `http://localhost:5173`

   **For network access (accessible from other machines):**
   ```bash
   npm run dev:network
   ```
   Then access from:
   - Your machine: `http://localhost:5173`
   - Other machines: `http://192.168.0.42:5173` (replace with your machine's IP)
   
   **Note:** The frontend will automatically detect the backend URL based on the hostname. When accessed via network IP, it will connect to the backend on the same IP address.

### Building for Production

```bash
npm run build
```

The production build will be in the `dist` directory.

### Preview Production Build

**For local access:**
```bash
npm run preview
```

**For network access:**
```bash
npm run preview:network
```
Then access from `http://192.168.0.42:4173` (replace with your machine's IP)

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

The frontend automatically detects the backend URL based on the hostname:
- **Local access** (`localhost` or `127.0.0.1`): Connects to `http://localhost:8001`
- **Network access** (e.g., `192.168.0.42`): Connects to `http://192.168.0.42:8001`

This ensures the frontend always connects to the backend on the same machine. You can modify the API configuration in `src/api/config.js` if needed.

### Network Access Setup

To allow other machines on your network to access the frontend:

1. **Start backend with network access:**
   - Ensure backend is running and accessible at `http://192.168.0.42:8001`
   - Backend CORS should allow requests from your network (already configured)

2. **Start frontend with network access:**
   ```bash
   npm run dev:network
   ```

3. **Access from other machines:**
   - Open browser on another machine
   - Navigate to `http://192.168.0.42:5173` (replace with your machine's IP)
   - The frontend will automatically connect to the backend at `http://192.168.0.42:8001`

**Security Note:** Network access exposes your dev server to your local network. Only use this in trusted network environments (home/office LAN). For production, use proper deployment with HTTPS and authentication.

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
