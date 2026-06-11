#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <juce_audio_utils/juce_audio_utils.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace
{
double nowSeconds()
{
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    return std::chrono::duration<double> (clock::now() - start).count();
}

std::runtime_error fail (const std::string& message)
{
    return std::runtime_error ("RenderKing: " + message);
}

std::string parameterStableId (juce::AudioProcessorParameter* parameter, int index)
{
    if (auto* withId = dynamic_cast<juce::AudioProcessorParameterWithID*> (parameter))
        if (withId->paramID.isNotEmpty())
            return withId->paramID.toStdString();

    return "param_" + std::to_string (index);
}

} // namespace

class RenderKingHost
{
public:
    RenderKingHost (const std::string& pluginPath,
                    int sampleRate,
                    int blockSize,
                    double renderDuration,
                    double tailDuration,
                    double warmupDuration,
                    int note,
                    int velocity)
        : pluginPath_ (pluginPath),
          sampleRate_ (sampleRate),
          blockSize_ (blockSize),
          renderDuration_ (renderDuration),
          tailDuration_ (tailDuration),
          warmupDuration_ (warmupDuration),
          note_ (note),
          velocity_ (velocity)
    {
        if (sampleRate_ <= 0)
            throw fail ("sample_rate must be positive.");
        if (blockSize_ <= 0)
            throw fail ("block_size must be positive.");
        if (renderDuration_ <= 0.0)
            throw fail ("render_duration must be positive.");
        if (note_ < 0 || note_ > 127)
            throw fail ("note must be in MIDI range 0-127.");
        if (velocity_ < 0 || velocity_ > 127)
            throw fail ("velocity must be in MIDI range 0-127.");

        juce::addDefaultFormatsToManager (formatManager_);
        loadPlugin();
        refreshParameterIds();
    }

    py::dict inspectPlugin() const
    {
        py::dict result;
        result["path"] = pluginPath_;
        result["name"] = processor_->getName().toStdString();
        result["version"] = processor_->getPluginDescription().version.toStdString();
        result["identifier"] = processor_->getPluginDescription().createIdentifierString().toStdString();
        result["is_instrument"] = processor_->isMidiEffect() || processor_->acceptsMidi();
        result["parameter_count"] = processor_->getParameters().size();
        result["program_controls"] = py::none();
        return result;
    }

    std::vector<py::dict> listParameters() const
    {
        std::vector<py::dict> result;
        const auto& parameters = processor_->getParameters();
        result.reserve ((size_t) parameters.size());

        for (int index = 0; index < parameters.size(); ++index)
        {
            auto* parameter = parameters.getUnchecked (index);
            py::dict item;
            item["stable_id"] = parameterIds_.at ((size_t) index);
            item["display_name"] = parameter->getName (128).toStdString();
            item["index"] = index;
            item["default_value"] = parameter->getDefaultValue();
            item["minimum"] = 0.0;
            item["maximum"] = 1.0;
            item["automatable"] = parameter->isAutomatable();
            item["is_meta"] = parameter->isMetaParameter();
            result.push_back (item);
        }

        return result;
    }

    void setParameters (const std::unordered_map<std::string, double>& normalizedValues)
    {
        const auto& parameters = processor_->getParameters();
        for (const auto& [stableId, normalizedValue] : normalizedValues)
        {
            const auto found = parameterIndexById_.find (stableId);
            if (found == parameterIndexById_.end())
                throw fail ("unknown parameter '" + stableId + "'.");

            const auto value = (float) std::clamp (normalizedValue, 0.0, 1.0);
            parameters.getUnchecked (found->second)->setValueNotifyingHost (value);
        }
    }

    std::unordered_map<std::string, double> currentParameterSnapshot() const
    {
        std::unordered_map<std::string, double> result;
        const auto& parameters = processor_->getParameters();
        for (int index = 0; index < parameters.size(); ++index)
            result[parameterIds_.at ((size_t) index)] = parameters.getUnchecked (index)->getValue();
        return result;
    }

    py::bytes capturePresetState() const
    {
        juce::MemoryBlock state;
        processor_->getStateInformation (state);
        return py::bytes (static_cast<const char*> (state.getData()), (py::ssize_t) state.getSize());
    }

    void restorePresetState (const py::bytes& bytes)
    {
        const std::string state = bytes;
        processor_->setStateInformation (state.data(), (int) state.size());
    }

    void selectProgram (int index)
    {
        const auto count = processor_->getNumPrograms();
        if (count <= 0)
            throw fail ("plugin does not expose programs.");
        if (index < 0 || index >= count)
            throw fail ("program index out of range.");
        processor_->setCurrentProgram (index);
    }

    std::vector<py::dict> enumerateProgramStates (py::object maxProgramsObject)
    {
        int maxPrograms = -1;
        if (! maxProgramsObject.is_none())
            maxPrograms = maxProgramsObject.cast<int>();

        std::vector<py::dict> states;
        const int programCount = processor_->getNumPrograms();
        if (programCount <= 1)
        {
            const auto state = capturePresetState();
            py::dict item;
            item["target_id"] = "state-default";
            item["label"] = "default_state";
            item["program_index"] = py::none();
            item["state_bytes"] = state;
            item["state_hash"] = "";
            states.push_back (item);
            return states;
        }

        const int limit = maxPrograms > 0 ? std::min (programCount, maxPrograms) : programCount;
        for (int index = 0; index < limit; ++index)
        {
            processor_->setCurrentProgram (index);
            const auto state = capturePresetState();
            py::dict item;
            item["target_id"] = "program-" + juce::String (index).paddedLeft ('0', 3).toStdString();
            item["label"] = processor_->getProgramName (index).toStdString();
            item["program_index"] = index;
            item["state_bytes"] = state;
            item["state_hash"] = "";
            states.push_back (item);
        }
        return states;
    }

    py::array_t<float> renderNote (py::object parametersObject, py::object noteObject, py::object durationObject, py::object velocityObject)
    {
        if (! parametersObject.is_none())
            setParameters (parametersObject.cast<std::unordered_map<std::string, double>>());

        const int note = noteObject.is_none() ? note_ : noteObject.cast<int>();
        const double duration = durationObject.is_none() ? renderDuration_ : durationObject.cast<double>();
        const int velocity = velocityObject.is_none() ? velocity_ : velocityObject.cast<int>();

        auto audio = render (note, duration, velocity);
        py::array_t<float> array ((py::ssize_t) audio.size());
        auto mutableArray = array.mutable_unchecked<1>();
        for (py::ssize_t index = 0; index < mutableArray.shape (0); ++index)
            mutableArray (index) = audio[(size_t) index];
        return array;
    }

    std::vector<py::dict> renderBatch (const std::vector<py::dict>& requests)
    {
        std::vector<py::dict> results;
        results.reserve (requests.size());

        for (const auto& request : requests)
        {
            const auto started = nowSeconds();
            const int slotId = request["slot_id"].cast<int>();
            const auto renderMode = request["render_mode"].cast<std::string>();

            if (renderMode == "target_state")
            {
                if (! request.contains ("preset_state") || request["preset_state"].is_none())
                    throw fail ("target_state render requires preset_state.");
                restorePresetState (request["preset_state"].cast<py::bytes>());
            }
            else if (renderMode == "parameter_state")
            {
                if (! request.contains ("parameters") || request["parameters"].is_none())
                    throw fail ("parameter_state render requires parameters.");
                setParameters (request["parameters"].cast<std::unordered_map<std::string, double>>());
            }
            else
            {
                throw fail ("unsupported render_mode '" + renderMode + "'.");
            }

            auto audio = render (note_, renderDuration_, velocity_);
            py::array_t<float> array ((py::ssize_t) audio.size());
            auto mutableArray = array.mutable_unchecked<1>();
            for (py::ssize_t index = 0; index < mutableArray.shape (0); ++index)
                mutableArray (index) = audio[(size_t) index];

            py::dict result;
            result["slot_id"] = slotId;
            result["worker_id"] = 0;
            result["audio"] = array;
            result["sample_rate"] = sampleRate_;
            result["render_seconds"] = nowSeconds() - started;
            results.push_back (result);
        }

        return results;
    }

private:
    void loadPlugin()
    {
        const juce::File pluginFile (pluginPath_);
        if (! pluginFile.exists())
            throw fail ("plugin path does not exist: " + pluginPath_);
        if (pluginFile.getFileExtension().toLowerCase() != ".vst3")
            throw fail ("expected a .vst3 plugin path: " + pluginPath_);

        juce::OwnedArray<juce::PluginDescription> descriptions;
        bool found = false;
        for (auto* format : formatManager_.getFormats())
        {
            if (format->getName() != "VST3")
                continue;
            found = knownPlugins_.scanAndAddFile (pluginFile.getFullPathName(), true, descriptions, *format);
            if (found)
                break;
        }

        if (! found || descriptions.isEmpty())
            throw fail ("could not scan VST3 plugin: " + pluginPath_);

        juce::String error;
        processor_ = formatManager_.createPluginInstance (*descriptions[0], sampleRate_, blockSize_, error);
        if (processor_ == nullptr)
            throw fail ("could not create plugin instance: " + error.toStdString());

        processor_->enableAllBuses();
        processor_->setPlayConfigDetails (0, std::max (1, processor_->getTotalNumOutputChannels()), sampleRate_, blockSize_);
        processor_->prepareToPlay (sampleRate_, blockSize_);
    }

    void refreshParameterIds()
    {
        parameterIds_.clear();
        parameterIndexById_.clear();
        const auto& parameters = processor_->getParameters();
        for (int index = 0; index < parameters.size(); ++index)
        {
            auto id = parameterStableId (parameters.getUnchecked (index), index);
            if (parameterIndexById_.find (id) != parameterIndexById_.end())
                id = id + "_" + std::to_string (index);
            parameterIds_.push_back (id);
            parameterIndexById_[id] = index;
        }
    }

    std::vector<float> render (int note, double duration, int velocity)
    {
        if (note < 0 || note > 127)
            throw fail ("note must be in MIDI range 0-127.");
        if (velocity < 0 || velocity > 127)
            throw fail ("velocity must be in MIDI range 0-127.");
        if (duration <= 0.0)
            throw fail ("duration must be positive.");

        processor_->reset();

        const int totalSamples = std::max (1, (int) std::ceil ((duration + tailDuration_ + warmupDuration_) * sampleRate_));
        const int noteOffSample = std::clamp ((int) std::round (duration * sampleRate_), 0, totalSamples - 1);
        const int channels = std::max (1, processor_->getTotalNumOutputChannels());
        std::vector<float> mono ((size_t) totalSamples, 0.0f);

        for (int offset = 0; offset < totalSamples; offset += blockSize_)
        {
            const int blockSamples = std::min (blockSize_, totalSamples - offset);
            juce::AudioBuffer<float> buffer (channels, blockSamples);
            buffer.clear();
            juce::MidiBuffer midi;

            if (offset == 0)
                midi.addEvent (juce::MidiMessage::noteOn (1, note, (juce::uint8) velocity), 0);
            if (noteOffSample >= offset && noteOffSample < offset + blockSamples)
                midi.addEvent (juce::MidiMessage::noteOff (1, note), noteOffSample - offset);

            processor_->processBlock (buffer, midi);

            for (int sample = 0; sample < blockSamples; ++sample)
            {
                double sum = 0.0;
                for (int channel = 0; channel < channels; ++channel)
                    sum += buffer.getSample (channel, sample);
                mono[(size_t) offset + (size_t) sample] = (float) (sum / channels);
            }
        }

        return mono;
    }

    juce::ScopedJuceInitialiser_GUI juceInitialiser_;
    std::string pluginPath_;
    int sampleRate_;
    int blockSize_;
    double renderDuration_;
    double tailDuration_;
    double warmupDuration_;
    int note_;
    int velocity_;
    juce::AudioPluginFormatManager formatManager_;
    juce::KnownPluginList knownPlugins_;
    std::unique_ptr<juce::AudioPluginInstance> processor_;
    std::vector<std::string> parameterIds_;
    std::unordered_map<std::string, int> parameterIndexById_;
};

PYBIND11_MODULE (_renderking, module)
{
    module.doc() = "RenderKing experimental VST3 rendering backend for rl-synth-programmer.";
    py::class_<RenderKingHost> (module, "Host")
        .def (py::init<const std::string&, int, int, double, double, double, int, int>(),
              py::arg ("plugin_path"),
              py::arg ("sample_rate") = 44100,
              py::arg ("block_size") = 512,
              py::arg ("render_duration") = 1.0,
              py::arg ("tail_duration") = 0.25,
              py::arg ("warmup_duration") = 0.0,
              py::arg ("note") = 60,
              py::arg ("velocity") = 100)
        .def ("inspect_plugin", &RenderKingHost::inspectPlugin)
        .def ("list_parameters", &RenderKingHost::listParameters)
        .def ("set_parameters", &RenderKingHost::setParameters)
        .def ("current_parameter_snapshot", &RenderKingHost::currentParameterSnapshot)
        .def ("capture_preset_state", &RenderKingHost::capturePresetState)
        .def ("restore_preset_state", &RenderKingHost::restorePresetState)
        .def ("select_program", &RenderKingHost::selectProgram)
        .def ("enumerate_program_states", &RenderKingHost::enumerateProgramStates, py::arg ("max_programs") = py::none())
        .def ("render_note", &RenderKingHost::renderNote,
              py::arg ("parameters") = py::none(),
              py::arg ("note") = py::none(),
              py::arg ("duration") = py::none(),
              py::arg ("velocity") = py::none())
        .def ("render_batch", &RenderKingHost::renderBatch);
}
