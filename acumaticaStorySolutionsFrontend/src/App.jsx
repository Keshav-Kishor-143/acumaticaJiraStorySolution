import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Alert,
  useTheme,
  Fade,
  Button,
  CircularProgress,
} from '@mui/material';
import {
  AutoStories as StoriesIcon,
  AutoAwesome as SparkleIcon,
} from '@mui/icons-material';
import toast, { Toaster } from 'react-hot-toast';
import StoryForm from './components/StoryForm';
import StreamingSolutionDisplay from './components/StreamingSolutionDisplay';
import CollapsedStorySummary from './components/CollapsedStorySummary';
import LoadingSpinner from './components/LoadingSpinner';
import HealthStatus from './components/HealthStatus';
import ManualsList from './components/ManualsList';
import { processStory, processStoryStream, handleApiError } from './api/solutions';

const App = () => {
  const theme = useTheme();
  const [isLoading, setIsLoading] = useState(false);
  const [solutionData, setSolutionData] = useState(null);
  const [error, setError] = useState(null);
  const [abortController, setAbortController] = useState(null);
  const [currentRequestId, setCurrentRequestId] = useState(null);
  const [streamingProgress, setStreamingProgress] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  
  // Layout transformation states
  const [isFormCollapsed, setIsFormCollapsed] = useState(false);
  const [isManualsExpanded, setIsManualsExpanded] = useState(true);
  const [submittedStoryData, setSubmittedStoryData] = useState(null);
  
  // Refs for auto-scroll
  const solutionAreaRef = useRef(null);
  const solutionContentRef = useRef(null);

  // Auto-minimize form when solution is generating or exists
  useEffect(() => {
    if ((solutionData || isLoading) && !isFormCollapsed) {
      setIsFormCollapsed(true);
      setIsManualsExpanded(false);
    }
  }, [solutionData, isLoading, isFormCollapsed]);

  const handleSubmit = async (storyData) => {
    // Store submitted data for collapsed view
    setSubmittedStoryData(storyData);
    
    // Trigger layout transformation - Solution takes priority
    setIsFormCollapsed(true);
    setIsManualsExpanded(false); // Auto-close manuals
    
    setIsLoading(true);
    setIsStreaming(true);
    setError(null);
    setSolutionData(null);
    setStreamingProgress(null);

    // Create new AbortController for this request
    const controller = new AbortController();
    setAbortController(controller);
    
    // Auto-scroll to solution area after layout transformation
    setTimeout(() => {
      if (solutionAreaRef.current) {
        solutionAreaRef.current.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'start',
          inline: 'nearest'
        });
      }
    }, 1200); // Wait for animation to complete (800ms transition + 400ms buffer)

    let accumulatedSolution = '';
    let finalSolutionData = null;

    try {
      // Use streaming for real-time updates
      await processStoryStream(
        storyData,
        // onChunk - called for each token
        (chunk) => {
          accumulatedSolution += chunk;
          // Update solution data in real-time
          setSolutionData(prev => ({
            ...prev,
            success: true,
            solution_markdown: accumulatedSolution,
            sources: prev?.sources || []
          }));
        },
        // onProgress - called for progress updates
        (progress) => {
          setStreamingProgress(progress);
        },
        // onComplete - called when done
        (completeData) => {
          finalSolutionData = {
            success: true,
            solution_markdown: completeData.solution || accumulatedSolution,
            sources: [],
            saved_file_path: completeData.saved_file_path,
            processing_time: completeData.processing_time
          };
          setSolutionData(finalSolutionData);
          setIsLoading(false);
          setIsStreaming(false);
          setStreamingProgress(null);
        },
        // onError
        (err) => {
          if (err.message === 'Request cancelled by user') {
            setError(null);
            setIsLoading(false);
            setIsStreaming(false);
          } else {
            const errorMessage = err.message || 'An error occurred while processing the story';
            setError(errorMessage);
            handleApiError(err);
            setIsLoading(false);
            setIsStreaming(false);
          }
        },
        controller
      );
    } catch (err) {
      // Fallback to non-streaming if streaming fails
      try {
        const response = await processStory(storyData, controller);
        
        if (response.request_id) {
          setCurrentRequestId(response.request_id);
        }
        
        if (response.success) {
          setSolutionData(response);
        } else {
          setError(response.error || 'Failed to generate solution');
          handleApiError(new Error(response.error || 'Failed to generate solution'));
        }
      } catch (fallbackErr) {
        if (fallbackErr.message === 'Request cancelled by user' || fallbackErr.message === 'Processing cancelled') {
          setError(null);
        } else {
          const errorMessage = fallbackErr.message || 'An error occurred while processing the story';
          setError(errorMessage);
          handleApiError(fallbackErr);
        }
      } finally {
        setIsLoading(false);
        setIsStreaming(false);
        setAbortController(null);
        setCurrentRequestId(null);
      }
    }
  };

  const handleCancel = async () => {
    if (abortController) {
      // Cancel the frontend request
      abortController.abort();
    }
    
    // Also cancel on backend if we have request_id
    if (currentRequestId) {
      try {
        const { cancelStoryProcessing } = await import('./api/solutions');
        await cancelStoryProcessing(currentRequestId);
        toast.success('Processing cancelled');
      } catch (err) {
        // Backend cancellation may fail if request already completed - that's okay
        console.log('Backend cancellation:', err.message);
      }
    }
    
    setIsLoading(false);
    setIsStreaming(false);
    setAbortController(null);
    setCurrentRequestId(null);
  };
  
  // Function to expand form again
  // IMPORTANT: Preserve solution state if solution is generating/exists
  const handleExpandForm = () => {
    // Only clear solution state if NOT currently loading/generating
    if (!isLoading && !isStreaming && !solutionData) {
      // No active solution - safe to reset everything
      setIsFormCollapsed(false);
      setIsManualsExpanded(true);
      setSubmittedStoryData(null);
      setSolutionData(null);
      setError(null);
      setIsLoading(false);
      setIsStreaming(false);
      setStreamingProgress(null);
    } else {
      // Solution is active or generating - just expand form but keep solution visible
      // Form will auto-minimize again due to useEffect above
      setIsFormCollapsed(false);
      setIsManualsExpanded(true);
      // Don't clear solution state - it will remain visible
    }
  };
  
  // Function to reset everything (for new story)
  const handleResetForm = () => {
    setIsFormCollapsed(false);
    setIsManualsExpanded(true);
    setSubmittedStoryData(null);
    setSolutionData(null);
    setError(null);
    setIsLoading(false);
    setIsStreaming(false);
    setStreamingProgress(null);
    setAbortController(null);
    setCurrentRequestId(null);
  };
  
  // Auto-scroll as solution streams (throttled to avoid excessive scrolling)
  useEffect(() => {
    if (isStreaming && solutionData?.solution_markdown) {
      // Throttle scroll updates to every 800ms for smoother experience
      const timer = setTimeout(() => {
        if (solutionContentRef.current) {
          solutionContentRef.current.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'end',
            inline: 'nearest'
          });
        }
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [solutionData?.solution_markdown, isStreaming]);

  return (
    <Box
      component="main"
      sx={{
        
        height: '100vh',
        width: '100%',
        bgcolor: 'background.default',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        position: 'relative',
        background: theme.palette.mode === 'dark' 
          ? 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)'
          : 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
        backgroundAttachment: 'fixed',
        backgroundSize: 'cover',
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

      {/* Modern Header */}
      <AppBar
        position="static"
        elevation={0}
        sx={{
          background: theme.palette.mode === 'dark'
            ? 'linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%)'
            : 'linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%)',
          borderBottom: `2px solid ${theme.palette.mode === 'dark' ? 'rgba(59, 130, 246, 0.2)' : 'rgba(59, 130, 246, 0.1)'}`,
          backdropFilter: 'blur(20px) saturate(180%)',
          boxShadow: theme.palette.mode === 'dark'
            ? '0 4px 20px rgba(0, 0, 0, 0.3)'
            : '0 4px 20px rgba(0, 0, 0, 0.05)',
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '2px',
            background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary?.main || theme.palette.primary.light}, ${theme.palette.primary.main})`,
            backgroundSize: '200% 100%',
            animation: 'shimmer 3s ease-in-out infinite',
            '@keyframes shimmer': {
              '0%': { backgroundPosition: '200% 0' },
              '100%': { backgroundPosition: '-200% 0' },
            },
          },
        }}
      >
        <Toolbar 
          sx={{ 
            minHeight: { xs: 60, sm: 68 },
            py: { xs: 0.75, sm: 1 },
            px: { xs: 2, sm: 2.5 },
            gap: 2.5,
          }}
        >
          {/* Logo & Title Section */}
          <Box 
            sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 2.5,
              flexShrink: 0,
            }}
          >
            <Box
              sx={{
                width: { xs: 44, sm: 48 },
                height: { xs: 44, sm: 48 },
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 50%, ${theme.palette.secondary?.main || theme.palette.primary.light} 100%)`,
                boxShadow: `0 8px 24px ${theme.palette.primary.main}40, 0 0 0 1px ${theme.palette.primary.main}20`,
                color: 'white',
                position: 'relative',
                overflow: 'hidden',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: '-50%',
                  left: '-50%',
                  width: '200%',
                  height: '200%',
                  background: 'radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%)',
                  animation: 'pulse 3s ease-in-out infinite',
                  '@keyframes pulse': {
                    '0%, 100%': { opacity: 0 },
                    '50%': { opacity: 1 },
                  },
                },
                '&:hover': {
                  transform: 'scale(1.05) rotate(5deg)',
                  boxShadow: `0 12px 32px ${theme.palette.primary.main}50`,
                },
              }}
            >
              <StoriesIcon sx={{ fontSize: { xs: 24, sm: 26 }, position: 'relative', zIndex: 1 }} />
            </Box>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
              <Typography
                variant="h5"
                component="div"
                sx={{
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1.5,
                  color: theme.palette.text.primary,
                  fontSize: { xs: '1.15rem', sm: '1.35rem' },
                  letterSpacing: '-0.02em',
                  background: theme.palette.mode === 'dark'
                    ? `linear-gradient(135deg, ${theme.palette.text.primary} 0%, ${theme.palette.primary.light} 100%)`
                    : `linear-gradient(135deg, ${theme.palette.text.primary} 0%, ${theme.palette.primary.main} 100%)`,
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                Story Solutions
                <SparkleIcon 
                  sx={{ 
                    fontSize: { xs: 18, sm: 22 },
                    color: theme.palette.primary.main,
                    animation: 'sparkle 2s ease-in-out infinite',
                    '@keyframes sparkle': {
                      '0%, 100%': { 
                        opacity: 1,
                        transform: 'scale(1) rotate(0deg)',
                      },
                      '50%': { 
                        opacity: 0.7,
                        transform: 'scale(1.2) rotate(180deg)',
                      },
                    },
                  }} 
                />
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  color: theme.palette.text.secondary,
                  display: { xs: 'none', sm: 'block' },
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  letterSpacing: '0.01em',
                }}
              >
                Generate comprehensive solutions for your Acumatica stories
              </Typography>
            </Box>
          </Box>

          {/* Spacer */}
          <Box sx={{ flex: 1 }} />

          {/* Health Status */}
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              px: 2,
              py: 1,
              borderRadius: 2,
              background: theme.palette.mode === 'dark'
                ? 'rgba(59, 130, 246, 0.1)'
                : 'rgba(59, 130, 246, 0.05)',
              border: `1px solid ${theme.palette.mode === 'dark' ? 'rgba(59, 130, 246, 0.2)' : 'rgba(59, 130, 246, 0.15)'}`,
              transition: 'all 0.3s ease',
              '&:hover': {
                background: theme.palette.mode === 'dark'
                  ? 'rgba(59, 130, 246, 0.15)'
                  : 'rgba(59, 130, 246, 0.08)',
                transform: 'translateY(-1px)',
              },
            }}
          >
            <HealthStatus />
          </Box>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Container
        maxWidth="xl"
        sx={{
          flex: 1,
          py: 1,
          px: 2,
          display: 'flex',
          gap: 1.5,
          overflow: 'hidden',
          minHeight: 0,
          height: '100%',
          maxHeight: '100%',
          transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
        }}
      >
        {/* Left Sidebar - Manuals List + Collapsed Story Summary */}
        <Box
          sx={{
            width: { xs: '100%', md: isFormCollapsed ? 280 : 320 },
            minWidth: { md: isFormCollapsed ? 280 : 280 },
            maxWidth: { md: isFormCollapsed ? 280 : 320 },
            display: { xs: 'none', md: 'flex' },
            flexDirection: 'column',
            gap: isFormCollapsed ? 2 : 0,
            height: '100%',
            maxHeight: '100%',
            minHeight: 0,
            transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
            flexShrink: 0,
            overflow: 'hidden',
          }}
        >
          {/* Collapsed Story Summary - Shown when form is collapsed OR solution is active */}
          {(isFormCollapsed || solutionData || isLoading) && submittedStoryData && (
            <Fade in={isFormCollapsed || solutionData || isLoading} timeout={800}>
              <Box>
                <CollapsedStorySummary
                  storyData={submittedStoryData}
                  onExpand={handleExpandForm}
                  onReset={handleResetForm}
                  isSolutionActive={!!(solutionData || isLoading)}
                />
              </Box>
            </Fade>
          )}
          
          {/* Manuals List */}
          <Box
            sx={{
              flex: isFormCollapsed ? '0 1 auto' : 1,
              minHeight: 0,
              maxHeight: '100%',
              height: isFormCollapsed ? 'auto' : '100%',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              transition: 'flex 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
            }}
          >
            <ManualsList 
              isExpanded={isManualsExpanded}
              onToggle={(expanded) => setIsManualsExpanded(expanded)}
              autoCollapse={isFormCollapsed}
            />
          </Box>
        </Box>

        {/* Main Content Area - Separated Sections */}
        <Box
          sx={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            minWidth: 0,
            gap: { xs: 2, sm: 2.5 },
            transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
            height: '100%',
            maxHeight: '100%',
            overflow: 'hidden',
          }}
        >
          {/* 
            PRIORITY SYSTEM: Solution takes priority when it exists or is generating
            - If solutionData exists OR isLoading: Show Solution section (form auto-minimizes)
            - If no solution and not loading: Show Story Form section
          */}
          
          {/* Story Form Section - Only shown when NO solution is active/generating */}
          {!solutionData && !isLoading && !isStreaming && (
            <Box
              sx={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                minHeight: 0,
                transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
                overflow: 'hidden',
              }}
            >
              <StoryForm 
                onSubmit={handleSubmit} 
                onCancel={handleCancel}
                isLoading={false} 
                canCancel={false}
              />
            </Box>
          )}

          {/* Loading Banner - Show when form is expanded but solution is generating */}
          {!isFormCollapsed && (isLoading || isStreaming) && (
            <Fade in={!isFormCollapsed && (isLoading || isStreaming)}>
              <Alert 
                severity="info" 
                sx={{ 
                  borderRadius: 2,
                  boxShadow: theme.shadows[2],
                  mb: 2,
                }}
                action={
                  <Button 
                    size="small" 
                    onClick={() => setIsFormCollapsed(true)}
                    sx={{ textTransform: 'none' }}
                  >
                    View Progress
                  </Button>
                }
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <CircularProgress size={20} />
                  <Typography variant="body2">
                    Generating solution in background... Click "View Progress" to see it.
                  </Typography>
                </Box>
              </Alert>
            </Fade>
          )}

          {/* Solution Section - Takes priority when solution exists or is generating */}
          {(solutionData || isLoading || isStreaming) && (
            <Box
              ref={solutionAreaRef}
              sx={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                minWidth: 0,
                gap: 3,
                transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
                overflowY: 'auto',
                overflowX: 'hidden',
                maxHeight: '100%',
                '&::-webkit-scrollbar': {
                  width: '10px',
                },
                '&::-webkit-scrollbar-track': {
                  background: 'transparent',
                },
                '&::-webkit-scrollbar-thumb': {
                  background: theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.2)' 
                    : 'rgba(0, 0, 0, 0.2)',
                  borderRadius: '5px',
                  '&:hover': {
                    background: theme.palette.mode === 'dark' 
                      ? 'rgba(255, 255, 255, 0.3)' 
                      : 'rgba(0, 0, 0, 0.3)',
                  },
                },
              }}
            >
              {/* Loading State - Show when generating solution */}
              {isLoading && !solutionData && (
                <LoadingSpinner 
                  message="Processing your story and generating solution..." 
                  progress={streamingProgress}
                />
              )}

              {/* Streaming Solution Display - Show when solution exists */}
              {solutionData && (
                <Box ref={solutionContentRef}>
                  <StreamingSolutionDisplay
                    solution={solutionData.solution_markdown}
                    storyId={solutionData.story_id}
                    processingTime={solutionData.processing_time}
                    savedFilePath={solutionData.saved_file_path}
                    isStreaming={isStreaming}
                    progress={streamingProgress}
                  />
                </Box>
              )}

              {/* Error State */}
              {error && !isLoading && (
                <Fade in={!!error}>
                  <Alert 
                    severity="error" 
                    sx={{ 
                      borderRadius: 2,
                      boxShadow: theme.shadows[2],
                    }}
                  >
                    {error}
                  </Alert>
                </Fade>
              )}
            </Box>
          )}
        </Box>
      </Container>
    </Box>
  );
};

export default App;
