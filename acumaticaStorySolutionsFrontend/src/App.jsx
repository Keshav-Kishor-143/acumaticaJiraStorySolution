import React, { useState, useEffect } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Alert,
  useTheme,
} from '@mui/material';
import {
  AutoStories as StoriesIcon,
  AutoAwesome as SparkleIcon,
} from '@mui/icons-material';
import { Toaster } from 'react-hot-toast';
import StoryForm from './components/StoryForm';
import SolutionDisplay from './components/SolutionDisplay';
import SourceReferences from './components/SourceReferences';
import LoadingSpinner from './components/LoadingSpinner';
import HealthStatus from './components/HealthStatus';
import { processStory, handleApiError } from './api/solutions';

const App = () => {
  const theme = useTheme();
  const [isLoading, setIsLoading] = useState(false);
  const [solutionData, setSolutionData] = useState(null);
  const [error, setError] = useState(null);
  const [healthStatus, setHealthStatus] = useState('checking');

  const handleSubmit = async (storyData) => {
    setIsLoading(true);
    setError(null);
    setSolutionData(null);

    try {
      const response = await processStory(storyData);
      
      if (response.success) {
        setSolutionData(response);
      } else {
        setError(response.error || 'Failed to generate solution');
        handleApiError(new Error(response.error || 'Failed to generate solution'));
      }
    } catch (err) {
      const errorMessage = err.message || 'An error occurred while processing the story';
      setError(errorMessage);
      handleApiError(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: 'background.default',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: theme.palette.background.paper,
            color: theme.palette.text.primary,
          },
        }}
      />

      {/* Header */}
      <AppBar
        position="static"
        elevation={0}
        sx={{
          bgcolor: theme.palette.background.paper,
          borderBottom: `1px solid ${theme.palette.divider}`,
          backdropFilter: 'blur(8px)',
          backgroundColor:
            theme.palette.mode === 'dark'
              ? 'rgba(30, 41, 59, 0.8)'
              : 'rgba(255, 255, 255, 0.9)',
        }}
      >
        <Toolbar sx={{ minHeight: { xs: 64, sm: 70 }, gap: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Box
              sx={{
                width: 42,
                height: 42,
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.primary.dark})`,
                boxShadow: `0 4px 12px ${theme.palette.primary.main}25`,
                color: 'white',
              }}
            >
              <StoriesIcon sx={{ fontSize: 24 }} />
            </Box>
            <Box>
              <Typography
                variant="h6"
                component="div"
                sx={{
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  color: theme.palette.text.primary,
                }}
              >
                Story Solutions
                <SparkleIcon sx={{ fontSize: 20, color: theme.palette.primary.main }} />
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  color: theme.palette.text.secondary,
                  display: { xs: 'none', sm: 'block' },
                }}
              >
                Generate comprehensive solutions for JIRA stories
              </Typography>
            </Box>
          </Box>

          <Box sx={{ flex: 1 }} />

          <HealthStatus />
        </Toolbar>
        {healthStatus === 'unhealthy' && (
          <Alert
            severity="error"
            sx={{
              borderRadius: 0,
              alignItems: 'center',
            }}
          >
            Backend service is not responding. Please check if the backend is running on port 8001.
          </Alert>
        )}
      </AppBar>

      {/* Main Content */}
      <Container
        maxWidth="lg"
        sx={{
          flex: 1,
          py: 4,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Story Form */}
        <StoryForm onSubmit={handleSubmit} isLoading={isLoading} />

        {/* Loading State */}
        {isLoading && (
          <Box sx={{ mt: 3 }}>
            <LoadingSpinner message="Processing your story and generating solution..." />
          </Box>
        )}

        {/* Error State */}
        {error && !isLoading && (
          <Alert severity="error" sx={{ mt: 3 }}>
            {error}
          </Alert>
        )}

        {/* Solution Display */}
        {solutionData && !isLoading && (
          <>
            <SolutionDisplay
              solution={solutionData.solution_markdown}
              storyId={solutionData.story_id}
              processingTime={solutionData.processing_time}
              savedFilePath={solutionData.saved_file_path}
            />

            {/* Source References */}
            {solutionData.sources && solutionData.sources.length > 0 && (
              <SourceReferences sources={solutionData.sources} />
            )}
          </>
        )}
      </Container>
    </Box>
  );
};

export default App;
