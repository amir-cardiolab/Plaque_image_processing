function varargout = plaque_quantification(varargin)
% PLAQUE_QUANTIFICATION MATLAB code for plaque_quantification.fig
%      PLAQUE_QUANTIFICATION, by itself, creates a new PLAQUE_QUANTIFICATION or raises the existing
%      singleton*.
%
%      H = PLAQUE_QUANTIFICATION returns the handle to a new PLAQUE_QUANTIFICATION or the handle to
%      the existing singleton*.
%
%      PLAQUE_QUANTIFICATION('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PLAQUE_QUANTIFICATION.M with the given input arguments.
%
%      PLAQUE_QUANTIFICATION('Property','Value',...) creates a new PLAQUE_QUANTIFICATION or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before plaque_quantification_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to plaque_quantification_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help plaque_quantification

% Last Modified by GUIDE v2.5 29-Jun-2020 20:39:23

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @plaque_quantification_OpeningFcn, ...
                   'gui_OutputFcn',  @plaque_quantification_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before plaque_quantification is made visible.
function plaque_quantification_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to plaque_quantification (see VARARGIN)

% Choose default command line output for plaque_quantification
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes plaque_quantification wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = plaque_quantification_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
